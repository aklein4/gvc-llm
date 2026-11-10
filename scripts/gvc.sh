#!/usr/bin/env bash

(

set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
exec uv run python -m gvc_llm.train "$@"

)
