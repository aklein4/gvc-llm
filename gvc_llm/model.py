"""The return-conditioned critic. Only LoRA and the prior train."""

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


NEGATIVE_PROMPT = "Deliberately make a mistake and give an incorrect final answer.\n\n"


class GenerativeCritic(nn.Module):
    def __init__(self, pretrained, rank=8, lm_head_chunk_size=128, *, tokenizer):
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
        self.register_buffer("negative_prompt_ids", torch.tensor(
            tokenizer.encode(NEGATIVE_PROMPT, add_special_tokens=False),
            device=embedding.device, dtype=torch.long,
        ))
        if not self.negative_prompt_ids.numel():
            raise ValueError("Negative prompt must tokenize to at least one token")
        self.prior = nn.Linear(embedding.shape[1], 2, device=embedding.device, dtype=torch.float32)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)

    @property
    def decoder(self):
        return self.lm.get_base_model().model  # Qwen2.5 decoder; LoRA modules remain active.

    @property
    def prefix_length(self):
        return self.negative_prompt_ids.numel()

    def features(self, ids, mask, returns, *, cache=None, use_cache=False,
                 position_ids=None, cache_position=None, prefill=None):
        # Reserve the same prefix slots for both branches; positive slots are
        # masked out so its positions and visible context remain unchanged.
        if prefill is None:
            prefill = cache is None or cache.get_seq_length() == 0
        width = ids.shape[1]
        n = self.prefix_length
        prefix_mask = (returns == 0)[:, None].expand(-1, n)
        offset = prefix_mask.sum(-1, keepdim=True)
        if position_ids is None:
            position_ids = positions(mask)[:, -width:]
        position_ids = position_ids + offset
        if mask.ndim == 2:
            mask = torch.cat((prefix_mask.to(mask.dtype), mask), dim=-1)
        else:
            # Static-cache masks include unused future keys. Add instruction
            # keys to every original query, and causal instruction queries.
            prefix_keys = prefix_mask[:, None, None, :].expand(-1, 1, width, -1)
            original_mask = mask
            mask = torch.cat((prefix_keys, mask), dim=-1)
            if prefill:
                prefix_queries = torch.zeros(
                    (ids.shape[0], 1, n, original_mask.shape[-1] + n),
                    dtype=mask.dtype, device=ids.device,
                )
                prefix_queries[..., :n] = (
                    torch.ones(n, n, dtype=torch.bool, device=ids.device).tril()[None, None]
                    & prefix_mask[:, None, None, :]
                )
                mask = torch.cat((prefix_queries, mask), dim=-2)
        if prefill:
            ids = torch.cat((self.negative_prompt_ids[None].expand(ids.shape[0], -1), ids), dim=-1)
            position_ids = torch.cat((positions(prefix_mask), position_ids), dim=-1)
            if cache_position is not None:
                cache_position = torch.arange(ids.shape[1], device=ids.device)
        elif cache_position is not None:
            cache_position = cache_position + n
        out = self.decoder(
            input_ids=ids, attention_mask=mask, position_ids=position_ids,
            past_key_values=cache, use_cache=use_cache, cache_position=cache_position,
        )
        return out.last_hidden_state[:, -width:], out.past_key_values

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
        Optional token-likelihood outcome weights use the full rollout batch.
        The prior is unweighted to preserve the observed return frequencies.
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
                                     labels[response_mask], reduction="sum")
        likelihood = self.token_nll_sum(observed[response_mask], actions[response_mask],
                                        outcome_weights[labels[response_mask]])
        return likelihood, prior_loss

    def optimizer(self, lora_lr=1e-4, head_lr=3e-4):
        return torch.optim.AdamW([
            {"params": [p for p in self.lm.parameters() if p.requires_grad], "lr": lora_lr},
            {"params": list(self.prior.parameters()), "lr": head_lr},
        ], weight_decay=0.0)
