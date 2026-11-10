"""Convert DeepMath's existing train/test splits to verl's reward schema."""
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

DATASET = "trl-lib/DeepMath-103K"
REVISION = "066c50a88d4e14cefc056e31111db2dba17f6c68"


def prepare_example(example, index, split):
    question = example["prompt"][0]["content"]
    return {
        "data_source": DATASET,
        "prompt": [{"role": "user", "content": f"Question:\n{question}\nAnswer:\nLet's think step by step.\n"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": example["solution"]},
        "extra_info": {"index": index, "split": split, "question": question},
    }


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    output = Path("data/deepmath")
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET, revision=REVISION)
    for split in ("train", "test"):
        prepared = dataset[split].map(
            prepare_example, with_indices=True, fn_kwargs={"split": split},
            remove_columns=dataset[split].column_names,
        )
        prepared.to_parquet(output / f"{split}.parquet")
        print(f"{split}: {len(prepared)} rows -> {output / (split + '.parquet')}")


if __name__ == "__main__":
    main()
