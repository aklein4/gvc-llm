"""The return-conditioned critic. Only LoRA, reward vectors, and prior train."""

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def positions(mask):
    return (mask.long().cumsum(-1) - 1).clamp_min(0)


def balanced_token_weights(returns, response_mask):
    """Full-batch weights: equal mass per present outcome, zero for absent ones."""
    counts = torch.stack([response_mask[returns == outcome].sum() for outcome in (0, 1)]).float()
    present = counts > 0
    return counts.sum() * present / (counts.clamp_min(1) * present.sum().clamp_min(1))


def posterior(logits0, logits1, prior_logits):
    """Bayes' rule over *vocabulary-normalized* branch likelihoods, in fp32."""
    log_p0 = logits0.float().log_softmax(-1)
    log_p1 = logits1.float().log_softmax(-1)
    log_prior = prior_logits.float().log_softmax(-1)
    return torch.sigmoid(log_p1 - log_p0 + log_prior[..., 1:2] - log_prior[..., 0:1])


class GenerativeCritic(nn.Module):
    def __init__(self, pretrained, rank=8, lm_head_chunk_size=128):
        super().__init__()
        if lm_head_chunk_size < 1:
            raise ValueError("lm_head_chunk_size must be positive")
        self.lm_head_chunk_size = lm_head_chunk_size
        self.lm = get_peft_model(pretrained, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
        embedding = self.lm.get_input_embeddings().weight
        # Population std across the vocabulary, separately for every dimension.
        initial = 0.1 * embedding.detach().float().std(dim=0, correction=0)
        self.reward_vectors = nn.Parameter(torch.stack((initial, torch.zeros_like(initial))))
        self.prior = nn.Linear(embedding.shape[1], 2, device=embedding.device, dtype=torch.float32)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

    @property
    def decoder(self):
        return self.lm.get_base_model().model  # Qwen2.5 decoder; LoRA modules remain active.

    def features(self, ids, mask, returns, *, cache=None, use_cache=False,
                 position_ids=None, cache_position=None):
        embeddings = self.lm.get_input_embeddings()(ids)
        embeddings = embeddings + self.reward_vectors[returns, None].to(embeddings.dtype)
        out = self.decoder(
            inputs_embeds=embeddings, attention_mask=mask,
            position_ids=positions(mask)[:, -ids.shape[1]:] if position_ids is None else position_ids,
            past_key_values=cache, use_cache=use_cache, cache_position=cache_position,
        )
        return out.last_hidden_state, out.past_key_values

    def logits(self, hidden):
        return self.lm.get_output_embeddings()(hidden)

    def _token_nll(self, hidden, targets, weights):
        return (F.cross_entropy(self.logits(hidden).float(), targets, reduction="none") * weights).sum()

    def token_nll_sum(self, hidden, targets, weights=None):
        # Checkpoint projection AND cross entropy so vocabulary-sized activations
        # are recomputed one chunk at a time during backward, not retained.
        loss = hidden.sum() * 0.0
        if weights is None:
            weights = torch.ones_like(targets, dtype=torch.float32)
        for h, y, w in zip(hidden.split(self.lm_head_chunk_size), targets.split(self.lm_head_chunk_size),
                           weights.split(self.lm_head_chunk_size)):
            loss = loss + checkpoint(self._token_nll, h, y, w, use_reentrant=False,
                                     preserve_rng_state=False)
        return loss

    def loss_sums(self, batch, outcome_weights=None):
        """Unnormalized response-token losses, for exact microbatch accumulation.

        State at position t predicts action at t+1. Prompt/pad tokens never
        contribute targets. Positive features train the prior for *all* returns.
        Optional outcome weights must be computed over the full rollout batch.
        """
        ids, mask = batch["input_ids"], batch["attention_mask"]
        returns, response_mask = batch["returns"].long(), batch["response_mask"].bool()
        length = response_mask.shape[1]
        actions = batch["responses"]
        # One decoder batch: positive conditioning for every prior target, plus
        # negative conditioning where required by the observed token likelihood.
        negative_rows = (returns == 0).nonzero(as_tuple=True)[0]
        hidden, _ = self.features(
            torch.cat((ids, ids[negative_rows])), torch.cat((mask, mask[negative_rows])),
            torch.cat((torch.ones_like(returns), returns[negative_rows])),
        )
        hidden = hidden[:, -length - 1:-1]
        positive = hidden[:ids.shape[0]]
        observed = positive.index_copy(0, negative_rows, hidden[ids.shape[0]:])
        labels = returns[:, None].expand_as(actions)
        if outcome_weights is None:
            outcome_weights = torch.ones(2, device=ids.device)
        prior_loss = F.cross_entropy(self.prior(positive[response_mask].float()),
                                     labels[response_mask], reduction="none")
        prior_loss = (prior_loss * outcome_weights[labels[response_mask]]).sum()
        likelihood = self.token_nll_sum(observed[response_mask], actions[response_mask],
                                        outcome_weights[labels[response_mask]])
        return likelihood, prior_loss

    def optimizer(self, lora_lr=1e-4, head_lr=3e-4):
        return torch.optim.AdamW([
            {"params": [p for p in self.lm.parameters() if p.requires_grad], "lr": lora_lr},
            {"params": [self.reward_vectors, *self.prior.parameters()], "lr": head_lr},
        ], weight_decay=0.0)
