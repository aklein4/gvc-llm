"""Measure fixed-length rollouts; separate compilation startup from warm throughput."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

from .model import GenerativeCritic
from .rollout import GuidedRollout


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-sizes', default='4,16')
    parser.add_argument('--modes', default='eager,static,compiled')
    parser.add_argument('--response-length', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', default='outputs/benchmark.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    set_seed(1)
    model = 'Qwen/Qwen2.5-0.5B'
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
    dataset = RLHFDataset('data/deepmath/train.parquet', tokenizer, OmegaConf.create({
        'max_prompt_length': 1024, 'filter_overlong_prompts_workers': 1,
        'apply_chat_template_kwargs': {'chat_template': '{{ messages[0].content }}'},
    }))
    policy = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).cuda().requires_grad_(False).eval()
    critic = GenerativeCritic(AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).cuda(), tokenizer=tokenizer).eval()
    # Only in this benchmark: ignore EOS so every mode does exactly B * length tokens.
    policy.generation_config.eos_token_id = -1
    results = []
    for size in map(int, args.batch_sizes.split(',')):
        prompts = DataProto.from_single_dict(collate_fn([dataset[i] for i in range(size)])).to('cuda')
        for mode in args.modes.split(','):
            rollout = GuidedRollout(policy, critic, tokenizer, response_length=args.response_length,
                                    decode_mode=mode, max_prompt_length=1024)
            times = []
            torch.cuda.reset_peak_memory_stats()
            for iteration in range(args.repeats + 1):
                set_seed(123)
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = rollout.generate_sequences(prompts)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                assert int(output.batch['response_mask'].sum()) == size * args.response_length
                del output
                print(f'{mode} batch={size} iteration={iteration}: {times[-1]:.3f}s', flush=True)
            row = dict(mode=mode, batch_size=size, response_length=args.response_length,
                       cold_seconds=times[0], warm_seconds=statistics.median(times[1:]),
                       tokens_per_second=size * args.response_length / statistics.median(times[1:]),
                       peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
            results.append(row)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            del rollout
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
