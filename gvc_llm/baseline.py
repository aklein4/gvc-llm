"""Load .env before handing control to verl's untouched GRPO/PPO entry point."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
# Ray should inherit this uv environment, not try to re-run uv with a new cwd.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

if __name__ == "__main__":
    if not any(arg.startswith("trainer.project_name=") for arg in sys.argv):
        sys.argv.insert(1, "trainer.project_name=" + os.getenv("WANDB_PROJECT", "gvc-llm"))
    from verl.trainer.main_ppo import main
    main()
