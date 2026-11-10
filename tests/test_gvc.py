"""Small CPU tests of the math, masking, branch gradients, and KV cache."""
import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl import DataProto
from gvc_llm.reward import compute_score

from gvc_llm.model import GenerativeCritic, posterior
from gvc_llm.rollout import GuidedRollout


def tiny_tokenizer():
    return SimpleNamespace(encode=lambda text, add_special_tokens=False: [11, 12, 13],
                           pad_token_id=0, eos_token_id=2)


def tiny_lm():
    torch.manual_seed(4)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=2, attention_dropout=0.0,
    ))


def tiny_batch(returns=(0, 1)):
    return {
        "input_ids": torch.tensor([[0, 3, 4, 5, 6], [7, 8, 9, 10, 0]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        "responses": torch.tensor([[5, 6], [10, 0]]),
        "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        "returns": torch.tensor(returns),
    }


def test_initialization_and_frozen_weights():
    lm = tiny_lm()
    critic = GenerativeCritic(lm, rank=2, tokenizer=tiny_tokenizer())
    assert not hasattr(critic, 'reward_vectors')
    assert not critic.negative_prompt_ids.requires_grad
    assert all('lora_' in n for n, p in critic.lm.named_parameters() if p.requires_grad)
    torch.testing.assert_close(critic.prior(torch.randn(3, 16)), torch.zeros(3, 2))
    opt = critic.optimizer(1e-4, 3e-4)
    assert [group['lr'] for group in opt.param_groups] == [1e-4, 3e-4]


def test_bayes_uses_vocab_normalization_and_prior():
    p0, p1 = torch.tensor([[.2, .8]]), torch.tensor([[.7, .3]])
    prior = torch.tensor([[.4, .6]])
    expected = p1 * .6 / (p1 * .6 + p0 * .4)
    torch.testing.assert_close(posterior(p0.log() + 50, p1.log() - 30, prior.log()), expected)
    torch.testing.assert_close(posterior(p0.log(), p0.log(), prior.log()), torch.full_like(p0, .6))


def test_loss_alignment_masking_and_positive_prior():
    critic = GenerativeCritic(tiny_lm(), rank=2, tokenizer=tiny_tokenizer())
    with torch.no_grad():
        critic.prior.weight.normal_(std=.1)
    batch = tiny_batch()
    nll, prior = critic.loss_sums(batch)
    expected_nll, expected_prior = 0, 0
    # Independent prefix-by-prefix calculation: only observed branch predicts action;
    # positive branch predicts the return even for negative-outcome trajectories.
    for row in range(2):
        for action_pos in range(2):
            if not batch['response_mask'][row, action_pos]:
                continue
            end = 3 + action_pos
            ids = batch['input_ids'][row:row+1, :end]
            mask = batch['attention_mask'][row:row+1, :end]
            r = batch['returns'][row:row+1]
            h, _ = critic.features(ids, mask, r)
            pos, _ = critic.features(ids, mask, torch.ones_like(r))
            expected_nll += -critic.logits(h[:, -1]).log_softmax(-1)[0, batch['responses'][row, action_pos]]
            expected_prior += -critic.prior(pos[:, -1]).log_softmax(-1)[0, r.item()]
    torch.testing.assert_close(nll, expected_nll)
    torch.testing.assert_close(prior, expected_prior)


def test_observed_branch_gradient_and_frozen_backbone():
    critic = GenerativeCritic(tiny_lm(), rank=2, tokenizer=tiny_tokenizer())
    batch = tiny_batch((0, 0))
    before = {n: p.detach().clone() for n, p in critic.named_parameters() if not p.requires_grad}
    nll, _ = critic.loss_sums(batch)
    nll.backward()
    assert any(p.grad is not None and p.grad.norm() > 0
               for p in critic.lm.parameters() if p.requires_grad)
    critic.zero_grad(set_to_none=True)
    # With a nonzero prior head, negative outcomes still train positive features.
    with torch.no_grad():
        critic.prior.weight.normal_()
    _, prior = critic.loss_sums(batch)
    prior.backward()
    assert any(p.grad is not None and p.grad.norm() > 0
               for p in critic.lm.parameters() if p.requires_grad)
    critic.optimizer().step()
    for n, p in critic.named_parameters():
        if n in before:
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)


def test_cached_conditioning_matches_full_prefix():
    critic = GenerativeCritic(tiny_lm(), rank=2, tokenizer=tiny_tokenizer()).eval()
    ids = torch.tensor([[0, 3, 4, 5], [6, 7, 8, 9]])
    mask = (ids != 0).long()
    r = torch.tensor([0, 1])
    with torch.no_grad():
        full, _ = critic.features(ids, mask, r)
        _, cache = critic.features(ids[:, :-1], mask[:, :-1], r, use_cache=True)
        incremental, _ = critic.features(ids[:, -1:], mask, r, cache=cache, use_cache=True)
    torch.testing.assert_close(full[:, -1], incremental[:, -1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("decode_mode", ["eager", "static"])
def test_prompt_conditioned_rollout_mask(decode_mode):
    base = tiny_lm().eval().requires_grad_(False)
    critic = GenerativeCritic(copy.deepcopy(base), rank=2, tokenizer=tiny_tokenizer()).eval()
    prompts = DataProto.from_dict(tensors={
        'input_ids': torch.tensor([[0, 3, 4], [6, 7, 8]]),
        'attention_mask': torch.tensor([[0, 1, 1], [1, 1, 1]]),
    })
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    torch.manual_seed(33)
    guided = GuidedRollout(base, critic, tokenizer, beta=.1, response_length=4, decode_mode=decode_mode, max_prompt_length=3).generate_sequences(prompts)
    torch.manual_seed(33)
    unguided = GuidedRollout(base, critic, tokenizer, beta=float('inf'), response_length=4, decode_mode=decode_mode, max_prompt_length=3).generate_sequences(prompts)
    assert 0 < guided.meta_info['metrics']['rollout/q_mean'] < 1
    assert unguided.meta_info['metrics']['rollout/kl'] == 0
    mask = guided.batch['response_mask']
    assert (mask[:, 1:].int() <= mask[:, :-1].int()).all()
    assert guided.batch['attention_mask'].shape == guided.batch['input_ids'].shape
    guided.batch['returns'] = torch.tensor([0, 1])
    nll, prior = critic.loss_sums(guided.batch)
    (nll + prior).backward()
    assert critic.prior.weight.grad.norm() > 0


def test_binary_math_reward_without_format_penalty():
    assert compute_score('simplerl', r'The answer is $\frac{1}{2}$.', '0.5') == 1
    assert compute_score('simplerl', r'The result is \boxed{42}.', '42') == 1
    assert compute_score('simplerl', 'The answer is 42.', '42') == 1
    assert compute_score('simplerl', 'The answer is 41.', '42') == 0
    assert compute_score('simplerl', 'I do not know.', '42') == 0


def test_microbatch_accumulation_preserves_token_weighting():
    batch = tiny_batch()  # Two response tokens in row 0; only one in row 1.
    full = GenerativeCritic(tiny_lm(), rank=2, tokenizer=tiny_tokenizer())
    micro = copy.deepcopy(full)
    denominator = batch['response_mask'].sum()
    nll, prior = full.loss_sums(batch)
    ((nll + prior) / denominator).backward()
    for row in range(2):
        nll, prior = micro.loss_sums({k: v[row:row+1] for k, v in batch.items()})
        ((nll + prior) / denominator).backward()
    for (name, p), (_, q) in zip(full.named_parameters(), micro.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=1e-4, msg=name)


def test_abel_prompt_is_not_wrapped_in_chat():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0}, unk_token='[UNK]')))
    prompt = "Question:\nWhat is 2+2?\nAnswer:\nLet's think step by step.\n"
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False,
        add_generation_prompt=True, chat_template='{{ messages[0].content }}',
    )
    assert rendered == prompt


def test_static_rollout_matches_dynamic_and_resets_after_update():
    base = tiny_lm().eval().requires_grad_(False)
    base.generation_config.eos_token_id = -1  # Fixed work for RNG-coupled comparison.
    critic = GenerativeCritic(copy.deepcopy(base), rank=2, tokenizer=tiny_tokenizer()).eval()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    prompts = DataProto.from_dict(tensors={
        'input_ids': torch.tensor([[0, 0, 3, 4], [0, 6, 7, 8]]),
        'attention_mask': torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
    })
    static = GuidedRollout(base, critic, tokenizer, response_length=4,
                           decode_mode='static', max_prompt_length=4)
    dynamic = GuidedRollout(base, critic, tokenizer, response_length=4)
    for update in range(2):
        torch.manual_seed(33)
        expected = dynamic.generate_sequences(prompts)
        torch.manual_seed(33)
        actual = static.generate_sequences(prompts)
        for key in expected.batch.keys():
            torch.testing.assert_close(actual.batch[key], expected.batch[key])
        assert actual.meta_info['metrics']['rollout/q_mean'] == pytest.approx(
            expected.meta_info['metrics']['rollout/q_mean'], abs=1e-6)
        # Make a real update between rollouts; cached features must not survive it.
        actual.batch['returns'] = torch.tensor([0, 1])
        critic.train()
        optimizer = critic.optimizer()
        optimizer.zero_grad()
        nll, prior = critic.loss_sums(actual.batch)
        (nll + prior).backward()
        optimizer.step()
        # Also change prefix length on the next call without changing cache capacity.
        prompts.batch['attention_mask'][0, 1] = 1


def test_delayed_stop_keeps_only_first_eos():
    base = tiny_lm().eval().requires_grad_(False)
    # Every vocabulary entry is a stop token: exactly one action per response.
    base.generation_config.eos_token_id = list(range(32))
    critic = GenerativeCritic(copy.deepcopy(base), rank=2, tokenizer=tiny_tokenizer()).eval()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    prompts = DataProto.from_dict(tensors={
        'input_ids': torch.tensor([[3, 4], [7, 8]]),
        'attention_mask': torch.ones(2, 2, dtype=torch.long),
    })
    actual = GuidedRollout(base, critic, tokenizer, response_length=8, decode_mode='static',
                           max_prompt_length=2, stop_check_interval=4).generate_sequences(prompts)
    assert actual.batch['responses'].shape == (2, 1)
    assert actual.batch['response_mask'].all()
    assert actual.meta_info['metrics']['rollout/truncated_fraction'] == 0


def test_deepmath_answers_and_schema():
    from gvc_llm.prepare_data import prepare_example

    row = prepare_example({'prompt': [{'role': 'user', 'content': 'What is 1/2?'}],
                           'solution': r'$\frac{1}{2}$'}, 7, 'test')
    assert row['reward_model']['ground_truth'] == r'$\frac{1}{2}$'
    assert row['extra_info']['split'] == 'test'
    assert row['extra_info']['index'] == 7
    assert r'\frac{1}{2}' not in row['prompt'][0]['content']
    assert compute_score(row['data_source'], r'The answer is $0.5$.', row['reward_model']['ground_truth']) == 1
    assert compute_score('deepmath', 'The answer is Yes.', 'Yes') == 1
    assert compute_score('deepmath', 'The answer is No.', 'Yes') == 0
    assert compute_score('deepmath', 'No', 'No') == 1
    assert compute_score('deepmath', 'I cannot determine the answer.', 'Yes') == 0


@pytest.mark.parametrize('returns', [(0, 1), (0, 0), (1, 1)])
def test_balanced_losses_and_microbatch_gradients(returns):
    from gvc_llm.model import balanced_token_weights

    batch = tiny_batch(returns)
    full = GenerativeCritic(tiny_lm(), rank=2, tokenizer=tiny_tokenizer())
    with torch.no_grad():
        full.prior.weight.normal_(std=.1)
    micro = copy.deepcopy(full)
    weights = balanced_token_weights(batch['returns'], batch['response_mask'])
    denominator = batch['response_mask'].sum()
    actual = full.loss_sums(batch, weights)
    # Token NLL averages outcome means; the prior averages all valid tokens.
    expected = torch.zeros(2)
    outcomes = set(returns)
    for outcome in outcomes:
        rows = batch['returns'] == outcome
        subset = {k: v[rows] for k, v in batch.items()}
        nll, prior = full.loss_sums(subset)
        expected = expected + torch.stack((nll / subset['response_mask'].sum() / len(outcomes),
                                           prior / denominator))
    torch.testing.assert_close(torch.stack(actual) / denominator, expected)
    (sum(actual) / denominator).backward()
    for row in range(2):
        losses = micro.loss_sums({k: v[row:row+1] for k, v in batch.items()}, weights)
        (sum(losses) / denominator).backward()
    for (name, p), (_, q) in zip(full.named_parameters(), micro.named_parameters()):
        if p.requires_grad:
            assert torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=1e-4, msg=name)
    for absent in {0, 1} - outcomes:
        assert weights[absent] == 0


def test_balanced_weights_with_no_response_tokens():
    from gvc_llm.model import balanced_token_weights

    weights = balanced_token_weights(torch.tensor([0, 1]), torch.zeros(2, 3, dtype=torch.bool))
    torch.testing.assert_close(weights, torch.zeros(2))


@pytest.mark.parametrize('chunk_size', [17, 128, 512])
def test_checkpointed_lm_head_matches_dense_loss_and_gradients(chunk_size):
    critic = GenerativeCritic(tiny_lm(), rank=2, lm_head_chunk_size=chunk_size, tokenizer=tiny_tokenizer())
    hidden = torch.randn(259, 16, requires_grad=True)  # Two full chunks and a tail.
    reference = hidden.detach().clone().requires_grad_(True)
    targets = torch.randint(32, (259,))
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = critic.token_nll_sum(hidden, targets)
    # No vocabulary-sized logits/log-softmax activations survive forward.
    assert not any(len(shape) == 2 and shape[-1] == 32 for shape in saved_shapes)
    expected = torch.nn.functional.cross_entropy(critic.logits(reference).float(), targets, reduction='sum')
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, reference.grad)


@pytest.mark.parametrize('returns', [(0, 1), (0, 0), (1, 1)])
def test_joint_training_forward_matches_separate_branches(returns):
    from gvc_llm.model import balanced_token_weights

    batch = tiny_batch(returns)
    joint = GenerativeCritic(tiny_lm(), rank=2, tokenizer=tiny_tokenizer())
    with torch.no_grad():
        joint.prior.weight.normal_(std=.1)
    separate = copy.deepcopy(joint)
    weights = balanced_token_weights(batch['returns'], batch['response_mask'])
    calls = []
    hook = joint.decoder.register_forward_pre_hook(lambda module, args: calls.append(1))
    actual = joint.loss_sums(batch, weights)
    hook.remove()
    assert len(calls) == 1
    expected_nll = expected_prior = 0
    for row, outcome in enumerate(returns):
        ids = batch['input_ids'][row:row+1]
        mask = batch['attention_mask'][row:row+1]
        selected = batch['response_mask'][row:row+1]
        r = batch['returns'][row:row+1]
        observed, _ = separate.features(ids, mask, r)
        positive, _ = separate.features(ids, mask, torch.ones_like(r))
        expected_nll += weights[outcome] * torch.nn.functional.cross_entropy(
            separate.logits(observed[:, -3:-1][selected]).float(),
            batch['responses'][row:row+1][selected], reduction='sum')
        expected_prior += torch.nn.functional.cross_entropy(
            separate.prior(positive[:, -3:-1][selected].float()),
            r.expand(int(selected.sum())), reduction='sum')
    torch.testing.assert_close(actual[0], expected_nll)
    torch.testing.assert_close(actual[1], expected_prior)
    sum(actual).backward()
    (expected_nll + expected_prior).backward()
    for (name, p), (_, q) in zip(joint.named_parameters(), separate.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=1e-4, msg=name)


@torch.no_grad()
def test_instruction_prefix_matches_explicit_prompt_and_preserves_positive():
    from gvc_llm.model import NEGATIVE_PROMPT, positions

    encoded = []
    tokenizer = tiny_tokenizer()
    tokenizer.encode = lambda text, add_special_tokens: encoded.append((text, add_special_tokens)) or [11, 12, 13]
    critic = GenerativeCritic(tiny_lm(), rank=2, tokenizer=tokenizer).eval()
    assert encoded == [(NEGATIVE_PROMPT, False)]
    ids = torch.tensor([[0, 3, 4], [5, 6, 7]])
    mask = (ids != 0).long()
    for outcome in (0, 1):
        actual, _ = critic.features(ids, mask, torch.full((2,), outcome))
        expected_ids, expected_mask = ids, mask
        if outcome == 0:
            expected_ids = torch.cat((torch.tensor([[11, 12, 13]]).expand(2, -1), ids), -1)
            expected_mask = torch.cat((torch.ones(2, 3, dtype=torch.long), mask), -1)
        expected = critic.decoder(input_ids=expected_ids, attention_mask=expected_mask,
                                  position_ids=positions(expected_mask)).last_hidden_state[:, -3:]
        torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=1e-6, rtol=1e-5)
