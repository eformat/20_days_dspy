## 20_days_of_dspy

https://gist.github.com/bllchmbrs/2a67e6ef278d003eecd3465a75603430
https://learnbybuilding.ai/20_days_of_dspy.zip

## setup

Python

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install --upgrade --quiet \
  mlflow \
  ipykernel \
  rich \
  dspy \
  ujson \
  openai \
  sentence_transformers \
  mcp
```

MLFLow

```bash
cd ~/git/dspy-tool-use
mlflow ui --port 5500 &
```

LLM's locally

```bash
export MODEL=DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf
export MODEL=Llama-3.2-3B-Instruct-Q8_0.gguf

podman run \
    -p 8080:8080 \
    --net=host \
    --device nvidia.com/gpu=0 \
    --security-opt label=type:nvidia_container_t \
    -v /home/mike/instructlab/models:/models:Z \
    ghcr.io/ggerganov/llama.cpp:full-cuda \
    --server -m /models/${MODEL} \
    --gpu-layers 999 \
    -np 3 \
    --ctx-size 18000
```

LLM's remotely

```bash
export LLM_URL=https://llama-4-scout-17b-16e-w4a16-your-maas:443/v1
export API_KEY=d2...
export LLM_MODEL="openai/llama-4-scout-17b-16e-w4a16"
```
