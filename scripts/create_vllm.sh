#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="/mnt/bn/yufei1900/wangxinhao/checkpoints/Qwen3.5-9B"
SERVED_MODEL_NAME="Qwen3.5-9B"
PORT="8000"
TENSOR_PARALLEL_SIZE="1"
MAX_MODEL_LEN="262144"
REASONING_PARSER="qwen3"
TOOL_CALL_PARSER="qwen3_coder"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  "$@"


