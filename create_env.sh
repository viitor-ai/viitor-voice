uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.12.0 --torch-backend=cu128
uv pip install -r requirements.txt
uv pip install dualcodec