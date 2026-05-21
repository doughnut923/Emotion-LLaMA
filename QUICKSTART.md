# Quickstart — Docker deployment (concise)

This quickstart shows the minimal steps to get the full Emotion‑LLaMA demo running in Docker while keeping large model files outside the build context.

Quick checklist (download/copy these before running):
- LLaMA weights: `checkpoints/Llama-2-7b-chat-hf/` (model shards + tokenizer + config)
- HuBERT audio encoder: `checkpoints/transformer/chinese-hubert-large/` (if not already present)
- Demo checkpoint (optional but recommended): `checkpoints/save_checkpoint/Emoation_LLaMA.pth`

Why we keep these outside the image
- These files are very large (tens of GB). Excluding them from the Docker build context makes image builds fast and prevents accidentally baking weights into the image. The repository's `.dockerignore` already excludes `checkpoints/` and similar folders.

1) Prepare model files (required)

- LLaMA (required for full functionality):
  - Minimal items the code expects in `checkpoints/Llama-2-7b-chat-hf/`:
    - `config.json`, `generation_config.json`, `pytorch_model.bin.index.json`
    - tokenizer: `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`
    - model shard files referenced by `pytorch_model.bin.index.json` (large `.bin` files)
  - If you have Hugging Face access, clone the model into the folder. If not, copy the files from another host or external drive into `checkpoints/Llama-2-7b-chat-hf/`.

- HuBERT audio encoder (for audio features):
  - Path: `checkpoints/transformer/chinese-hubert-large/` — the repo may already contain these files; otherwise download from the model provider and place them there.

- Demo fine-tuned checkpoint (optional):
  - `checkpoints/save_checkpoint/Emoation_LLaMA.pth` — if present, the demo will load the fine‑tuned weights.

2) Build the image (fast because checkpoints are ignored)

Use BuildKit for faster builds and caching (optional):

```bash
DOCKER_BUILDKIT=1 docker build -t emotion-llama:latest .
```

3) Choose how the container gets checkpoints

Default option: use the model files baked into the image by the HF download step. This is the simplest path and does not require mounts.

```bash
docker-compose up --build
```

Local-mount option: if you want to use checkpoint files from your host machine instead, use the override file `docker-compose.local.yml`.

```bash
docker-compose -f docker-compose.yml -f docker-compose.local.yml up --build
```

For the local-mount option, the expected host layout is:
- `./checkpoints/Llama-2-7b-chat-hf/`
- `./transformer/chinese-hubert-large/`
- `./save_checkpoint/Emoation_LLaMA.pth`

Example `docker run` (adjust host paths):

```bash
# single container
docker run --gpus all \
  -v /full/path/to/Emotion-LLaMA/checkpoints:/app/checkpoints:ro \
  -v /full/path/to/Emotion-LLaMA/transformer:/app/transformer:ro \
  -p 7860:7860 -p 7889:7889 \
  -e MODEL_PATH=/app/checkpoints/Llama-2-7b-chat-hf \
  -e HUBERT_PATH=/app/checkpoints/transformer/chinese-hubert-large \
  -e CKPT_PATH=/app/checkpoints/save_checkpoint/Emoation_LLaMA.pth \
  emotion-llama:latest

# or use docker-compose with the local-mount override:
docker-compose -f docker-compose.yml -f docker-compose.local.yml up --build
```

Notes:
- The default compose file now assumes the image already contains the downloaded models.
- Use `docker-compose.local.yml` only when you want the host folders to override the image files.
- `app.py` serves the main Gradio demo on port `7860`. `app_EmotionLlamaClient.py` runs an alternate client on port `7889`.

4) Run locally without Docker (optional)

If you prefer to run natively (for debugging), create the environment and install extras:

```bash
conda env create -f environment.yaml
conda activate llama
pip install moviepy==1.0.3 soundfile==0.12.1 opencv-python==4.7.0.72
```

Then ensure the same `checkpoints/` layout exists and run:

```bash
python app.py            # main demo (7860)
# or
python app_EmotionLlamaClient.py   # client demo (7889)
```

5) How to fetch models if you have Hugging Face access

```bash
huggingface-cli login
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf checkpoints/Llama-2-7b-chat-hf
```

If you lack HF access, copy the files manually from a host with the weights or an external drive into `checkpoints/Llama-2-7b-chat-hf/` (preserve filenames and folder layout).

6) Quick API examples

Curl (file path accessible to server):
```bash
curl -X POST "http://127.0.0.1:7860/api/predict/" \
  -H "Content-Type: application/json" \
  -d '{"data": ["/path/to/video.mp4", "Analyze the emotions in this video."]}'
```

Python (base64 video):
```python
import requests, base64
with open('/path/to/video.mp4','rb') as f:
    b64 = base64.b64encode(f.read()).decode()
payload = {"data": [f"data:video/mp4;base64,{b64}", "What emotions are shown?"]}
resp = requests.post('http://127.0.0.1:7860/api/predict/', json=payload, timeout=300)
print(resp.json())
```

Troubleshooting
- If the container errors at start: confirm you mounted `checkpoints/` into `/app/checkpoints` and that `MODEL_PATH` points to a valid LLaMA folder. The entrypoint will print helpful instructions.
- If model loading fails with missing shards, open `checkpoints/Llama-2-7b-chat-hf/pytorch_model.bin.index.json` and ensure the shard filenames listed there exist in the same folder.
- For CUDA/driver issues: ensure your host NVIDIA driver supports CUDA 11.7 and that the NVIDIA Container Toolkit is installed.

Need help wiring your specific host paths into `docker run` or `docker-compose.yml`? Tell me the host path where your model files live and I will produce the exact `docker run` and `docker-compose` snippets.
