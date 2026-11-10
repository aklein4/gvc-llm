# GVC and GRPO on DeepMath-103K

[Token-level GVC](description.md) and a runnable
[SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason) GRPO baseline,
using **Qwen/Qwen2.5-0.5B (base)**, **[trl-lib/DeepMath-103K](https://huggingface.co/datasets/trl-lib/DeepMath-103K)**, and **Abel prompts**.
The model and prompt style follow SimpleRL; the dataset is DeepMath-103K.
Both trainers use a binary math-correctness reward, with no format penalty.

## Run

One uv environment, one lockfile, no extras:

```bash
uv sync --python 3.10
uv run python -m gvc_llm.prepare_data

# Three real updates each, sized for a quick single-GPU test.
bash scripts/smoke_gvc.sh
bash scripts/smoke_baseline.sh

# Full-length runs (8192-token responses; substantially slower).
uv run python -m gvc_llm.train --steps 100
bash scripts/baseline.sh

uv run pytest -q
```

Both trainers read the repo's `.env`: `WANDB_API_KEY`, `WANDB_ENTITY`, optional
`WANDB_PROJECT` (default `gvc-llm`), and `HF_TOKEN`. Use `WANDB_MODE=offline` to
log locally. Existing environment variables take precedence. No secrets are
stored in the code or lockfile.

Overrides go last:

```bash
bash scripts/smoke_gvc.sh --steps 5 --beta 0.2 --head-lr 0.0003
bash scripts/smoke_baseline.sh trainer.total_training_steps=5
```

## Recipe

`prepare_data.py` converts the official DeepMath train/test splits at a pinned
revision to verl parquet files under `data/deepmath`. The user prompt is wrapped
in the same Abel `Question:/Answer:` format, and `solution` becomes the verified
ground truth. The supplied splits contain 97,870 training and 5,152 test rows
before prompt-length filtering. No new train/test split is made. Both trainers tokenize the prepared
prompt verbatim. The verifier also handles DeepMath's Yes/No answers.

The baseline follows the upstream
[launcher](https://github.com/hkust-nlp/simpleRL-reason/blob/cf1c7858bfc145a27aa733e75d930d34f0d318a6/train_grpo_math_tune_ray.sh)
and the paper's main-run settings: GRPO, 1024 prompts/update, 8 responses/prompt,
8192 response tokens, minibatch 256, LR `5e-7`, clip `0.2`, temperature `1`,
actor KL coefficient `1e-4` (`low_var_kl`), entropy coefficient `0.001`, and
20 epochs. KL is an actor loss, not a reward penalty. There is no scalar critic.

This is a small port to verl 0.6.1, not a vendored copy of SimpleRL's old fork.
It uses one GPU/TP=1, small log-probability microbatches, local Ray startup,
and disables loss compilation. Verification uses verl's installed Math-Verify
integration rather than vendoring SimpleRL's older Qwen answer extractor; the
0/1 correctness rule is shared by both trainers, though extraction edge cases
can differ. The smoke baseline uses 2 prompts × 8 responses, a 512-token limit,
64 training examples, eager vLLM, and no evaluation/checkpointing.

GVC uses a local PyTorch loop with verl's dataset, `DataProto`, and logging;
the baseline uses verl's GRPO trainer and vLLM. GVC's smoke test uses 16 responses
per update and a 512-token limit. Its full run defaults to 16 responses/update;
it has no GRPO grouping. Neither smoke test measures final model quality.

## GVC implementation

- `model.py`: frozen pretrained critic + shared LoRA, two reward vectors,
  positive-conditioned binary prior, token loss, and Bayesian Q calculation.
- `rollout.py`: frozen-policy generation with static KV caches, compiled decoding,
  and preallocated buffers. Logits are `(base_logits + Q / beta) / T`.
- `train.py`: generate, verify, train the critic, log, and save.
- `reward.py`: the shared correctness verifier.

The negative reward vector starts at `0.1 * std(embedding, dim=0, correction=0)`;
the positive vector starts at zero. They are added to every prompt/response
token embedding. The prior starts at zero. Only the
observed-return branch receives token-likelihood targets; the positive branch
supplies prior features for both outcomes. Prompt/padding targets are masked,
EOS is included. Token NLL gives equal weight to the positive
and negative outcome token means, using counts from the complete rollout batch
across all microbatches. If only one outcome exists, its token mean gets full
weight; the absent outcome contributes zero. Prior NLL is unweighted and averaged
over all valid response tokens, preserving their observed return frequencies.
Branch probabilities are vocabulary-normalized
before Bayes' rule. The policy remains frozen throughout.

Training computes the LM head and cross entropy in checkpointed chunks of
`--lm-head-chunk-size` valid response tokens (default 128). This is the total
across sequences and both outcomes within a microbatch, not per sequence.
Both conditioned branches share one decoder forward pass; outcome balancing
is applied to the per-token losses. Positive features are reused for the prior.
Backward recomputes each chunk's logits, trading extra
compute for lower activation memory while preserving the loss and gradients.

Defaults: LoRA rank 8/LR `1e-4`, reward-vector/prior LR `3e-4`, beta `0.1`.
`--beta inf` samples the unguided frozen policy. All settings are in `train.py`'s
small dataclass. There is no policy-gradient, TD, or value-anchor loss in GVC.

GVC saves config, JSONL metrics/samples, and `critic.pt` under `outputs/gvc`
(`outputs/gvc-smoke` for smoke tests). The checkpoint holds trainable parameters,
optimizer state, config, and step; construct the same critic and load
`checkpoint['trainable']` with `strict=False`. There is no exact RNG/data-loader
resume. Reusing an output directory overwrites that run's files.

The locked environment uses PyTorch 2.8, vLLM 0.11, Transformers 4.57.6, and
FlashAttention 2.8.3. On this ARM GH200, uv selects CUDA 12.9 wheels (CUDA 12.8
on x86). FlashAttention was installed from the local uv cache; a fresh machine
may require the CUDA toolkit to build it. GVC uses PyTorch SDPA.

## Faster sampling

GVC defaults to `--decode-mode compiled` and batch size 16. Prefill runs eagerly;
the one-token decode graph compiles both model paths, vocabulary heads, the
FP32 Bayesian posterior, and sampling with `torch.compile(mode="reduce-overhead")`.
Both critic branches remain batched together. SDPA and the existing BF16/FP32
dtypes are preserved; no explicit FA2 or TF32 precision change is introduced.

Static caches reserve `max_prompt_length + max_response_length` positions. They
reset between rollouts, so each update uses the current critic weights. Buffers
are preallocated and position IDs increment in place. Completion is checked
every 16 tokens (`--stop-check-interval`); any extra padding after EOS is removed
and never contributes a training target. Compilation can change floating-point
rounding and RNG consumption, so identical seeds need not produce identical text.

Use `--decode-mode eager` for the original dynamic-cache implementation, or
`--decode-mode static` for the buffer/cache changes without compilation.
Changing batch size or cache capacity can require a new compiled graph. The
first rollout includes compilation; later updates reuse it. Larger batches
increase throughput and also change the number of samples per optimizer update.

```bash
# Same prompts and fixed output lengths for each mode; EOS ignored only here.
TORCHINDUCTOR_COMPILE_THREADS=4 uv run python -m gvc_llm.benchmark

# Compare training paths at the same batch size.
bash scripts/smoke_gvc.sh --decode-mode eager --batch-size 16
bash scripts/smoke_gvc.sh --decode-mode compiled --batch-size 16
```

The following measurements used the previous SimpleRL data; the benchmark
command now uses DeepMath. Measured on this GH200, generating exactly 128 tokens
per response; warm times
are medians of two runs after one warmup. Results are in `outputs/benchmark.json`.

| Mode | Batch | Warm rollout time | Generated tokens/s |
|---|---:|---:|---:|
| Original eager | 4 | 10.33 s | 50 |
| Static, uncompiled | 4 | 9.33 s | 55 |
| Compiled | 4 | 1.14 s | 449 |
| Original eager | 16 | 10.66 s | 192 |
| Static, uncompiled | 16 | 9.64 s | 213 |
| Compiled | 16 | 1.56 s | 1,316 |

The first compiled rollouts took about 38 s including startup. Compilation gave
9.1× speedup at batch 4 and 6.8× at batch 16. Combining compilation with batch 16
increased token throughput about 26.5× over the original batch-4 configuration;
that comparison includes the larger batch, not just compiler improvements.

## DeepMath validation

14 tests pass, including DeepMath schema and answer scoring checks. A GVC smoke
run completed one optimizer update with 16 responses and a 512-token limit in
57 s including compilation warmup, using 8.9 GB peak allocated GPU memory.
No responses received positive rewards; this validates execution, not accuracy.
Checkpoint: `outputs/gvc-deepmath-smoke/critic.pt`.
[W&B run](https://wandb.ai/outer-loop/gvc-llm/runs/v84s7qyy).
The GRPO configuration points to DeepMath; its training run below used the
previous dataset.

## Previous performance validation (SimpleRL data)

13 tests pass, including CUDA comparisons of compiled critic outputs against full
prefix evaluation before and after weight changes, cache reset, and EOS masking.

The compiled GVC smoke run completed three optimizer updates with 16 responses
per update and a 512-token limit. The first step took 50 s including compilation;
the next two took 12.5–12.6 s, with about 7 s sampling and 7.2 GB peak allocated
memory. Two trajectories received positive returns. This is an execution check,
not evidence of improved accuracy. The checkpoint is
`outputs/gvc-compiled-smoke/critic.pt`.

The GRPO smoke baseline previously completed three updates at 15–17 s/update.
Its 16 responses are grouped as 2 prompts × 8, while GVC uses 16 distinct prompts.

W&B: [compiled GVC smoke](https://wandb.ai/outer-loop/gvc-llm/runs/l1v64657),
[GRPO smoke](https://wandb.ai/outer-loop/gvc-llm/runs/1l7fiv0m).
