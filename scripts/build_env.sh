uv venv --python 3.10
source .venv/bin/activate

uv sync --extra vllm --extra gui --extra rag --extra code_interpreter --extra mcp
