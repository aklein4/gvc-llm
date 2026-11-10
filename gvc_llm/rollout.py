"""Single-GPU, KV-cached guided rollout with verl DataProto inputs/outputs."""

import math

import torch
from verl import DataProto
from transformers import StaticCache

from .model import posterior, positions


class GuidedRollout:
    def __init__(self, policy, critic, tokenizer, beta=0.01, temperature=1.0, response_length=512,
                 decode_mode="eager", max_prompt_length=1024, stop_check_interval=16):
        if beta <= 0 or temperature <= 0 or response_length <= 0:
            raise ValueError("beta, temperature, and response_length must be positive")
        self.policy, self.critic, self.tokenizer = policy, critic, tokenizer
        self.beta, self.temperature, self.response_length = beta, temperature, response_length
        self.guided = math.isfinite(beta)
        # In-place updates preserve the compiled graph and CUDA graph input address.
        self._beta = torch.tensor(beta, device=next(policy.parameters()).device)
        if decode_mode not in ("eager", "static", "compiled") or stop_check_interval < 1:
            raise ValueError("decode_mode must be eager/static/compiled; stop_check_interval must be positive")
        self.decode_mode = decode_mode
        self.capacity = max_prompt_length + response_length
        self.stop_check_interval = stop_check_interval
        self.base_cache = self.critic_cache = None
        self.cache_batch_size = None
        # Prefill stays eager. Only fixed-shape one-token decoding is compiled.
        # This graph includes both critics, LM heads, posterior, and sampling.
        self.decode = (
            torch.compile(self._step, mode="reduce-overhead", fullgraph=True, dynamic=False)
            if decode_mode == "compiled" else self._step
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if self.decode_mode == "eager":
            return self._generate_eager(prompts)
        return self._generate_static(prompts)

    def _generate_eager(self, prompts: DataProto) -> DataProto:
        self.policy.eval()
        self.critic.eval()
        ids = prompts.batch["input_ids"]
        mask = prompts.batch["attention_mask"]
        # RLHFDataset pads prompts to the configured limit; remove common left padding.
        width = int(mask.sum(-1).max())
        ids, mask = ids[:, -width:], mask[:, -width:]
        prompt_ids = ids
        batch_size = ids.shape[0]
        branch_returns = torch.arange(2, device=ids.device).repeat_interleave(batch_size)
        base_cache = critic_cache = None
        alive = torch.ones(batch_size, dtype=torch.bool, device=ids.device)
        responses, response_masks, sampled_q, token_kls = [], [], [], []
        eos = self.policy.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        eos = torch.tensor(eos or [self.tokenizer.eos_token_id], device=ids.device)
        for _ in range(self.response_length):
            base = self.policy.model(
                input_ids=ids, attention_mask=mask,
                position_ids=positions(mask)[:, -ids.shape[1]:],
                past_key_values=base_cache, use_cache=True,
            )
            base_cache = base.past_key_values
            logits = self.policy.lm_head(base.last_hidden_state[:, -1]).float()
            base_log_probs = (logits / self.temperature).log_softmax(-1)
            if math.isfinite(self.beta):
                hidden, critic_cache = self.critic.features(
                    ids.repeat(2, 1), mask.repeat(2, 1), branch_returns,
                    cache=critic_cache, use_cache=True,
                )
                negative, positive = hidden[:, -1].chunk(2)
                q = posterior(self.critic.logits(negative), self.critic.logits(positive),
                              self.critic.prior(positive.float()))
                logits = logits + q / self.beta
            log_probs = (logits / self.temperature).log_softmax(-1)
            token_kls.append((log_probs.exp() * (log_probs - base_log_probs)).sum(-1))
            token = torch.multinomial(log_probs.exp(), 1).squeeze(1)
            token = torch.where(alive, token, self.tokenizer.pad_token_id)
            responses.append(token)
            response_masks.append(alive.clone())  # Include the first EOS as an action.
            if math.isfinite(self.beta):
                sampled_q.append(q.gather(1, token[:, None]).squeeze(1))
            alive = alive & ~(token[:, None] == eos).any(-1)
            mask = torch.cat([mask, response_masks[-1][:, None]], dim=1)
            ids = token[:, None]
            if not alive.any():
                break
        responses = torch.stack(responses, dim=1)
        response_mask = torch.stack(response_masks, dim=1)
        full_ids = torch.cat([prompt_ids, responses], dim=1)
        metrics = {"rollout/truncated_fraction": alive.float().mean().item()}
        metrics["rollout/kl"] = torch.stack(token_kls, dim=1)[response_mask].mean().item()
        if sampled_q:
            metrics["rollout/q_mean"] = torch.stack(sampled_q, dim=1)[response_mask].mean().item()
        return DataProto.from_dict(tensors={
            "prompts": prompt_ids, "responses": responses, "input_ids": full_ids,
            "attention_mask": mask, "position_ids": positions(mask),
            "response_mask": response_mask,
        }, non_tensors=prompts.non_tensor_batch, meta_info={"metrics": metrics})

    def _step(self, ids, mask, position_ids, cache_position, returns, prefill=False):
        """Eager prefill or compiled decoding; the mask already excludes future keys."""
        base = self.policy.model(
            input_ids=ids, attention_mask=mask, position_ids=position_ids,
            past_key_values=self.base_cache, cache_position=cache_position, use_cache=True,
        )
        logits = self.policy.lm_head(base.last_hidden_state[:, -1]).float()
        base_log_probs = (logits / self.temperature).log_softmax(-1)
        q = torch.zeros_like(logits)
        if self.guided:
            hidden, _ = self.critic.features(
                ids.repeat(2, 1), mask.repeat(2, 1, 1, 1), returns,
                position_ids=position_ids.repeat(2, 1), cache_position=cache_position,
                cache=self.critic_cache, use_cache=True, prefill=prefill,
            )
            negative, positive = hidden[:, -1].chunk(2)
            q = posterior(self.critic.logits(negative), self.critic.logits(positive),
                          self.critic.prior(positive.float()))
            logits = logits + q / self._beta
        log_probs = (logits / self.temperature).log_softmax(-1)
        kl = (log_probs.exp() * (log_probs - base_log_probs)).sum(-1)
        token = torch.multinomial(log_probs.exp(), 1)
        return token, q.gather(1, token), kl

    def _generate_static(self, prompts: DataProto) -> DataProto:
        self.policy.eval()
        self.critic.eval()
        original_ids, original_mask = prompts.batch["input_ids"], prompts.batch["attention_mask"]
        self._beta.fill_(self.beta)
        batch_size, device = original_ids.shape[0], original_ids.device
        width = int(original_mask.sum(-1).max())
        if width + self.response_length > self.capacity:
            raise ValueError("Prompt exceeds the configured static cache capacity")
        if self.cache_batch_size != batch_size:
            self.base_cache = StaticCache(self.policy.config, max_cache_len=self.capacity)
            self.critic_cache = StaticCache(self.critic.lm.config, max_cache_len=self.capacity + self.critic.prefix_length)
            self.cache_batch_size = batch_size
        else:
            self.base_cache.reset()
            self.critic_cache.reset()

        # Fixed backing storage, reused throughout decoding. No growing concatenations.
        ids = torch.full((batch_size, self.capacity), self.tokenizer.pad_token_id,
                         device=device, dtype=torch.long)
        ids[:, :width] = original_ids[:, -width:]
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[:, :width] = original_mask[:, -width:].bool()
        response_mask = torch.zeros((batch_size, self.response_length), device=device, dtype=torch.bool)
        sampled_q = torch.zeros_like(response_mask, dtype=torch.float32)
        token_kls = torch.zeros_like(sampled_q)
        returns = torch.arange(2, device=device).repeat_interleave(batch_size)
        alive = torch.ones(batch_size, dtype=torch.bool, device=device)
        eos = self.policy.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        eos = torch.tensor(eos or [self.tokenizer.eos_token_id], device=device)

        prefix_positions = positions(mask[:, :width])
        prefix_cache_positions = torch.arange(width, device=device)
        causal = torch.arange(self.capacity, device=device)[None, :] <= prefix_cache_positions[:, None]
        prefix_mask = mask[:, None, None, :] & causal[None, None, :, :]
        token, q, kl = self._step(ids[:, :width], prefix_mask, prefix_positions, prefix_cache_positions, returns, prefill=True)
        # Stable input shapes/strides for all one-token calls, independent of prompt length.
        next_ids = torch.empty((batch_size, 1), device=device, dtype=torch.long)
        next_positions = prefix_positions[:, -1:].contiguous()
        cache_position = torch.tensor([width - 1], device=device)
        decode_mask = mask[:, None, None, :]
        for t in range(self.response_length):
            if t:
                torch.compiler.cudagraph_mark_step_begin()
                token, q, kl = self.decode(next_ids, decode_mask, next_positions, cache_position, returns)
            next_ids.copy_(torch.where(alive[:, None], token, self.tokenizer.pad_token_id))
            ids[:, width + t].copy_(next_ids[:, 0])
            response_mask[:, t].copy_(alive)  # Includes EOS; later padding is never a target.
            sampled_q[:, t].copy_(q[:, 0])
            token_kls[:, t].copy_(kl)
            mask[:, width + t].copy_(alive)
            next_positions.add_(alive[:, None])
            cache_position.add_(1)
            alive.logical_and_(~(next_ids == eos).any(-1))
            # A little padding work avoids synchronizing the CPU after every token.
            if (t + 1) % self.stop_check_interval == 0 and not alive.any():
                break

        length = int(response_mask.sum(-1).max())
        response_mask = response_mask[:, :length].clone()
        full_ids = ids[:, :width + length].contiguous()
        full_mask = mask[:, :width + length].long()
        metrics = {"rollout/truncated_fraction": alive.float().mean().item()}
        metrics["rollout/kl"] = token_kls[:, :length][response_mask].mean().item()
        if math.isfinite(self.beta):
            metrics["rollout/q_mean"] = sampled_q[:, :length][response_mask].mean().item()
        return DataProto.from_dict(tensors={
            "prompts": full_ids[:, :width], "responses": full_ids[:, width:], "input_ids": full_ids,
            "attention_mask": full_mask, "position_ids": positions(full_mask),
            "response_mask": response_mask,
        }, non_tensors=prompts.non_tensor_batch, meta_info={"metrics": metrics})
