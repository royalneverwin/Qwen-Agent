#!/usr/bin/env bash

# faster: UV_LOCK_MODE=check bash scripts/build_env.sh
# fastest: UV_LOCK_MODE=frozen bash scripts/build_env.sh
# extra: PYTHON_VERSION=3.10 QWEN_AGENT_EXTRAS="vllm,gui,rag,code_interpreter,mcp" bash scripts/build_env.sh

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
UV_LOCK_MODE="${UV_LOCK_MODE:-update}"
QWEN_AGENT_EXTRAS="${QWEN_AGENT_EXTRAS:-vllm gui rag code_interpreter mcp}"

sync_args=(--python "${PYTHON_VERSION}" --no-dev)
for extra in ${QWEN_AGENT_EXTRAS//,/ }; do
    sync_args+=(--extra "${extra}")
done

case "${UV_LOCK_MODE}" in
    update)
        uv lock --python "${PYTHON_VERSION}"
        uv sync --locked "${sync_args[@]}"
        ;;
    check)
        uv lock --check --python "${PYTHON_VERSION}"
        uv sync --locked "${sync_args[@]}"
        ;;
    frozen)
        uv sync --frozen "${sync_args[@]}"
        ;;
    *)
        echo "Unsupported UV_LOCK_MODE='${UV_LOCK_MODE}'. Use update, check, or frozen." >&2
        exit 2
        ;;
esac
