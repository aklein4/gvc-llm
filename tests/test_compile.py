"""CUDA numerical check for the compiled conditioned decoder and Bayes calculation."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, StaticCache

from gvc_llm.model import GenerativeCritic, positions, posterior


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA compilation')
@torch.no_grad()
def test_compiled_critic_matches_full_prefix_after_weight_change():
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        pad_token_id=0, eos_token_id=2, attention_dropout=0.0,
    )).to(device='cuda', dtype=torch.bfloat16)
    critic = GenerativeCritic(model, rank=2).eval()
    critic.prior.weight.normal_(std=.1)
    for name, parameter in critic.lm.named_parameters():
        if 'lora_B' in name:
            parameter.normal_(std=.01)
    cache = StaticCache(critic.lm.config, max_cache_len=8)
    ids = torch.tensor([[0, 3, 4, 5], [0, 3, 4, 5]], device='cuda')
    mask = (ids != 0).long()
    returns = torch.tensor([0, 1], device='cuda')
    next_ids = ids[:, -1:].contiguous()
    next_positions = positions(mask)[:, -1:].contiguous()
    cache_position = torch.tensor([3], device='cuda')
    decode_mask = torch.zeros((2, 1, 1, 8), device='cuda', dtype=torch.bool)
    decode_mask[:, :, :, :4] = mask[:, None, None].bool()

    def decode():
        hidden, _ = critic.features(
            next_ids, decode_mask, returns, cache=cache, use_cache=True,
            position_ids=next_positions, cache_position=cache_position,
        )
        logits = critic.logits(hidden[:, -1])
        prior = critic.prior(hidden[:, -1].float())
        q = posterior(logits[:1], logits[1:], prior[1:])
        return logits, prior, q

    compiled = torch.compile(decode, mode='reduce-overhead', fullgraph=True)
    for update in range(2):
        cache.reset()
        critic.features(ids[:, :-1], mask[:, :-1], returns, cache=cache, use_cache=True)
        full, _ = critic.features(ids, mask, returns)
        logits = critic.logits(full[:, -1])
        prior = critic.prior(full[:, -1].float())
        expected = (logits, prior, posterior(logits[:1], logits[1:], prior[1:]))
        torch.compiler.cudagraph_mark_step_begin()
        actual = compiled()
        for a, b in zip(actual, expected):
            assert a.dtype == b.dtype
            torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-2)
        # In-place changes, as performed by AdamW, must be visible to compiled graphs.
        critic.reward_vectors.add_(torch.randn_like(critic.reward_vectors) * .01)
        for name, parameter in critic.lm.named_parameters():
            if 'lora_B' in name:
                parameter.add_(torch.randn_like(parameter) * .01)
