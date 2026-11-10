import copy
import math
from types import SimpleNamespace

import pytest
import torch
from verl import DataProto

from gvc_llm.adaptive_beta import AdaptiveBeta
from gvc_llm.model import GenerativeCritic, posterior
from gvc_llm.rollout import GuidedRollout
from test_gvc import tiny_lm, tiny_tokenizer


def test_beta_controller_defaults_and_strict_boundaries():
    controller = AdaptiveBeta()
    assert controller.beta == pytest.approx(0.01)
    assert controller.log_beta == pytest.approx(math.log2(0.01))
    for kl in (0.9, 1.0, 1.1):
        metrics = controller.update(kl)
        assert metrics['train/beta_increment'] == metrics['train/beta_decrement'] == 0
        assert controller.beta == pytest.approx(0.01)
    metrics = controller.update(1.1001)
    assert metrics['train/beta_increment'] == 1
    assert metrics['train/beta_decrement'] == 0
    assert metrics['train/log_beta'] == pytest.approx(math.log2(0.01) + 0.1)
    assert metrics['train/beta'] == pytest.approx(0.01 * 2 ** 0.1)
    metrics = controller.update(0.8999)
    assert metrics['train/beta_increment'] == 0
    assert metrics['train/beta_decrement'] == 1
    assert controller.beta == pytest.approx(0.01)


def test_beta_controller_configuration_and_unguided_mode():
    controller = AdaptiveBeta(beta=0.2, kl_target=2, kl_range=0.5, delta_log_beta=1)
    controller.update(3)
    assert controller.beta == pytest.approx(0.4)
    controller.update(1)
    assert controller.beta == pytest.approx(0.2)
    controller = AdaptiveBeta(beta=float('inf'))
    metrics = controller.update(0)
    assert controller.beta == float('inf')
    assert metrics['train/beta_increment'] == metrics['train/beta_decrement'] == 0


def test_beta_bounds_clamp_overshoots_and_allow_moving_back():
    controller = AdaptiveBeta(beta=0.02, min_beta=0.01, max_beta=0.03, delta_log_beta=2)
    metrics = controller.update(2)
    assert controller.beta == pytest.approx(0.03)
    assert controller.log_beta == math.log2(0.03)
    assert metrics['train/beta_increment'] == 1
    assert controller.update(2)['train/beta_increment'] == 0
    metrics = controller.update(0)
    assert controller.beta == pytest.approx(0.01)
    assert controller.log_beta == math.log2(0.01)
    assert metrics['train/beta_decrement'] == 1
    assert controller.update(0)['train/beta_decrement'] == 0
    assert controller.update(2)['train/beta_increment'] == 1


def test_default_beta_bounds_hold_over_repeated_updates():
    controller = AdaptiveBeta()
    for kl in [0] * 200 + [2] * 500:
        controller.update(kl)
        assert 0.0001 <= controller.beta <= 100


@pytest.mark.parametrize('kwargs', [
    {'beta': 0}, {'beta': -1}, {'beta': float('nan')},
    {'kl_target': -1}, {'kl_target': float('inf')},
    {'kl_range': -1}, {'kl_range': 2}, {'kl_range': float('nan')},
    {'delta_log_beta': 0}, {'delta_log_beta': float('inf')},
    {'min_beta': 0}, {'max_beta': float('inf')}, {'min_beta': float('nan')},
    {'min_beta': 2, 'max_beta': 1}, {'beta': 0.00001}, {'beta': 101},
])
def test_invalid_beta_configuration(kwargs):
    with pytest.raises(ValueError):
        AdaptiveBeta(**kwargs)


@pytest.mark.parametrize('decode_mode', ['eager', 'static', 'compiled'])
@torch.no_grad()
def test_rollout_kl_is_exact_and_token_weighted_after_beta_change(decode_mode, monkeypatch):
    base = tiny_lm().eval().requires_grad_(False)
    critic = GenerativeCritic(copy.deepcopy(base), rank=2, tokenizer=tiny_tokenizer()).eval()
    critic.prior.weight.normal_(std=0.1)
    prompts = DataProto.from_dict(tensors={
        'input_ids': torch.tensor([[0, 3, 4], [6, 7, 8]]),
        'attention_mask': torch.tensor([[0, 1, 1], [1, 1, 1]]),
    })
    rollout = GuidedRollout(base, critic, SimpleNamespace(pad_token_id=0, eos_token_id=2),
                            beta=0.01, temperature=0.7, response_length=4,
                            decode_mode='static' if decode_mode == 'compiled' else decode_mode,
                            max_prompt_length=3, stop_check_interval=4)
    compilations = []
    if decode_mode == 'compiled':
        # Exercise fullgraph capture and guards on CPU without a costly CUDA build.
        def backend(graph, inputs):
            compilations.append(graph)
            return graph.forward
        rollout.decode = torch.compile(rollout._step, backend=backend, fullgraph=True)

    # Different sequence lengths, including EOS and extra static padding steps.
    # Force the first row to EOS and use argmax for the second row, keeping
    # sampling deterministic without Python state inside the compiled graph.
    def sample(probs, count):
        token = probs.argmax(-1, keepdim=True)
        return torch.where(torch.arange(probs.shape[0])[:, None] == 0, 2, token)
    monkeypatch.setattr(torch, 'multinomial', sample)
    for beta in (0.01, 0.05):
        rollout.beta = beta
        batch = rollout.generate_sequences(prompts).batch
        expected = []
        for row in range(2):
            for t in range(batch['responses'].shape[1]):
                if not batch['response_mask'][row, t]:
                    continue
                end = 3 + t
                ids = batch['input_ids'][row:row+1, :end]
                mask = batch['attention_mask'][row:row+1, :end]
                positions = (mask.cumsum(-1) - 1).clamp_min(0)
                logits = base(input_ids=ids, attention_mask=mask, position_ids=positions).logits[:, -1].float()
                hidden, _ = critic.features(ids.repeat(2, 1), mask.repeat(2, 1), torch.tensor([0, 1]))
                negative, positive = hidden[:, -1].chunk(2)
                log_base = logits.log_softmax(-1)
                log_adjustment = critic.logits(negative).float().log_softmax(-1)
                log_negative = (2 * log_base - log_adjustment).log_softmax(-1)
                q = posterior(log_negative, critic.logits(positive), critic.prior(positive.float()))
                p = torch.distributions.Categorical(logits=logits / 0.7)
                tilted = torch.distributions.Categorical(logits=(logits + q / beta) / 0.7)
                expected.append(torch.distributions.kl_divergence(tilted, p))
        # Repeating deterministic generation also checks cache reset.
        metrics = rollout.generate_sequences(prompts).meta_info['metrics']
        assert metrics['rollout/kl'] == pytest.approx(torch.cat(expected).mean().item(), abs=2e-5)
        assert batch['response_mask'][0].sum() == 1
    if decode_mode == 'compiled':
        assert len(compilations) == 1  # Beta updates must not specialize a new graph.
