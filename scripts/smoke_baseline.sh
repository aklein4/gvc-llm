#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/baseline.sh" \
  data.train_batch_size=2 \
  data.val_batch_size=8 \
  data.train_max_samples=64 \
  data.val_max_samples=8 \
  data.max_response_length=512 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
  actor_rollout_ref.rollout.enforce_eager=True \
  trainer.total_training_steps=3 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.experiment_name=deepmath-grpo-smoke \
  trainer.rollout_data_dir=outputs/baseline-smoke \
  "$@"
