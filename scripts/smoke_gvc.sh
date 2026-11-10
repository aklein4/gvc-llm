#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
exec uv run python -m gvc_llm.train --steps 3 --batch-size 16 \
  --max-response-length 512 --name gvc-smoke --output-dir outputs/gvc-smoke "$@"
