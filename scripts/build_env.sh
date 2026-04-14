uv venv --python 3.10
source .venv/bin/activate

uv pip install --torch-backend=auto -e '.[vllm,gui,rag,code_interpreter,mcp]'
