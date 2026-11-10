#!/usr/bin/env bash
# SimpleRL-Zoo Qwen2.5-0.5B / DeepMath / Abel, ported to current verl on one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
exec uv run python -m gvc_llm.baseline \
  data.train_files=data/deepmath/train.parquet \
  data.val_files=data/deepmath/test.parquet \
  data.train_batch_size=1024 \
  data.val_batch_size=500 \
  data.max_prompt_length=1024 \
  data.max_response_length=8192 \
  '+data.apply_chat_template_kwargs.chat_template="{{ messages[0].content }}"' \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.0001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=10216 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path=gvc_llm/reward.py \
  custom_reward_function.name=compute_score \
  'trainer.logger=[console,wandb]' \
  trainer.experiment_name=deepmath-grpo-qwen-0.5b \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.total_epochs=20 \
  ray_kwargs.ray_init.num_cpus=8 \
  "$@"
