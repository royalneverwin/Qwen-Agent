#!/usr/bin/env bash

# build uv.lock: UV_LOCK_MODE=update bash scripts/build_env.sh
# faster: UV_LOCK_MODE=check bash scripts/build_env.sh
# fastest: UV_LOCK_MODE=frozen bash scripts/build_env.sh
# extra: PYTHON_VERSION=3.10 QWEN_AGENT_EXTRAS="vllm,gui,rag,code_interpreter,mcp" bash scripts/build_env.sh

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
UV_LOCK_MODE="${UV_LOCK_MODE:-frozen}"
QWEN_AGENT_EXTRAS="${QWEN_AGENT_EXTRAS:-vllm gui rag code_interpreter mcp}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.9}"

sync_args=(--python "${PYTHON_VERSION}" --no-dev)
for extra in ${QWEN_AGENT_EXTRAS//,/ }; do
    sync_args+=(--extra "${extra}")
done

lock_args=(--python "${PYTHON_VERSION}")
torch_packages=(vllm torch torchvision torchaudio)

case "${UV_LOCK_MODE}" in
    update)
        for package in "${torch_packages[@]}"; do
            lock_args+=(--upgrade-package "${package}")
        done
        uv lock "${lock_args[@]}"
        uv sync --locked "${sync_args[@]}"
        ;;
    check)
        uv lock --check "${lock_args[@]}"
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

.venv/bin/python -c 'import sys, torch; expected = sys.argv[1]; actual = torch.version.cuda; print(f"torch: {torch.__version__}, torch cuda: {actual}"); sys.exit(0 if actual == expected else f"Expected torch CUDA {expected}, got {actual}")' "${EXPECTED_TORCH_CUDA}"
source .venv/bin/activate