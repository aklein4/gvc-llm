"""Minimal on-policy GVC loop; no PPO loss, value bootstrap, or policy updates."""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from .reward import compute_score
from verl.utils.tracking import Tracking

from .model import GenerativeCritic
from .rollout import GuidedRollout
from .adaptive_beta import AdaptiveBeta


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B"
    train_file: str = "data/deepmath/train.parquet"
    steps: int = 1000
    batch_size: int = 32
    micro_batch_size: int = 8
    decode_mode: str = "compiled"
    stop_check_interval: int = 16
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    beta: float = 0.01
    kl_target: float = 1.0
    kl_range: float = 0.1
    delta_log_beta: float = 0.1
    min_beta: float = 0.0001
    max_beta: float = 100.0
    temperature: float = 1.0
    lora_rank: int = 32
    lm_head_chunk_size: int = 2048
    lora_lr: float = 1e-4
    head_lr: float = 3e-4
    seed: int = 1
    project: str = "gvc-llm"
    name: str = "gvc"
    output_dir: str = "outputs/gvc"
    save_every: int = 10


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config(project=os.getenv("WANDB_PROJECT", "gvc-llm"))
    for key, value in asdict(defaults).items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    cfg = Config(**vars(parser.parse_args()))
    if min(cfg.steps, cfg.batch_size, cfg.micro_batch_size, cfg.save_every, cfg.lm_head_chunk_size) < 1:
        parser.error("steps, batch sizes, save-every, and lm-head-chunk-size must be positive")
    try:
        beta_controller = AdaptiveBeta(cfg.beta, cfg.kl_target, cfg.kl_range, cfg.delta_log_beta,
                                       cfg.min_beta, cfg.max_beta)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(4)
    set_seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    logger = Tracking(cfg.project, cfg.name, ["console", "wandb"],
                      config={"trainer": {}, "gvc": asdict(cfg)})
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
    dataset = RLHFDataset(cfg.train_file, tokenizer, OmegaConf.create({
        "max_prompt_length": cfg.max_prompt_length,
        "filter_overlong_prompts_workers": 1,
        "truncation": "error",
        "apply_chat_template_kwargs": {"chat_template": "{{ messages[0].content }}"},
    }))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)
    if not len(loader):
        raise ValueError("Dataset is smaller than batch-size")
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().requires_grad_(False).eval()
    critic = GenerativeCritic(AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda(), cfg.lora_rank, lm_head_chunk_size=cfg.lm_head_chunk_size, tokenizer=tokenizer)
    optimizer = critic.optimizer(cfg.lora_lr, cfg.head_lr)
    rollout = GuidedRollout(policy, critic, tokenizer, beta_controller.beta, cfg.temperature, cfg.max_response_length,
                            cfg.decode_mode, cfg.max_prompt_length, cfg.stop_check_interval)
    if cfg.decode_mode == "compiled":
        print("First rollout includes torch.compile warmup; later rollouts reuse the graph.", flush=True)
    trainable = {n: p for n, p in critic.named_parameters() if p.requires_grad}
    print(f"Trainable critic parameters: {sum(p.numel() for p in trainable.values()):,}", flush=True)
    iterator = iter(loader)
    for step in range(1, cfg.steps + 1):
        start = time.perf_counter()
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        prompts = DataProto.from_single_dict(raw).to("cuda")
        batch = rollout.generate_sequences(prompts)
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - start
        texts = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        rewards = [compute_score("trl-lib/DeepMath-103K", text, info["ground_truth"])
                   for text, info in zip(texts, batch.non_tensor_batch["reward_model"])]
        batch.batch["returns"] = torch.tensor(rewards, device="cuda", dtype=torch.long)
        tokens = batch.batch["response_mask"].sum().item()
        critic.train()
        optimizer.zero_grad(set_to_none=True)
        likelihood_sum = prior_sum = 0.0
        for micro in batch.split(cfg.micro_batch_size):
            nll, prior = critic.loss_sums(micro.batch)
            ((nll + prior) / tokens).backward()
            likelihood_sum += nll.detach().item()
            prior_sum += prior.detach().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0).item()
        optimizer.step()
        beta_metrics = beta_controller.update(batch.meta_info["metrics"]["rollout/kl"])
        rollout.beta = beta_controller.beta
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        metrics = {
            **batch.meta_info["metrics"],
            **beta_metrics,
            "train/loss": (likelihood_sum + prior_sum) / tokens,
            "train/token_nll": likelihood_sum / tokens,
            "train/prior_nll": prior_sum / tokens,
            "train/grad_norm": grad_norm,
            "train/lora_lr": cfg.lora_lr, "train/head_lr": cfg.head_lr,
            "train/reward_mean": sum(rewards) / len(rewards),
            "train/positive_count": sum(rewards),
            "rollout/response_length_mean": tokens / len(rewards),
            "timing/rollout_seconds": generation_seconds, "timing/step_seconds": elapsed,
            "throughput/response_tokens_per_second": tokens / generation_seconds,
            "memory/peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        logger.log(metrics, step=step)
        with (output / "metrics.jsonl").open("w" if step == 1 else "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        with (output / "samples.jsonl").open("w" if step == 1 else "a") as f:
            for index, text, reward in zip(batch.non_tensor_batch["index"], texts, rewards):
                f.write(json.dumps({"step": step, "index": int(index), "response": text, "reward": reward}) + "\n")
        if step % cfg.save_every == 0 or step == cfg.steps:
            torch.save({
                "step": step, "config": asdict(cfg),
                "trainable": {n: p.detach().cpu() for n, p in trainable.items()},
                "optimizer": optimizer.state_dict(),
                "log_beta": beta_controller.log_beta,
            }, output / "critic.pt")
    # Explicit finish so the command exits only after W&B flushes.
    logger.logger["wandb"].finish()


if __name__ == "__main__":
    main()
