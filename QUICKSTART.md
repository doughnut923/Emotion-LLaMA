# Quickstart — Docker setup

This guide shows how to run Emotion-LLaMA in Docker, which model files are required, and the CUDA/runtime expectations.

**Summary**
- Base image: `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime` (CUDA 11.7)
- Server entrypoint: `app.py` (served with Gradio on port `7860` by default)
- Alternate demo: `app_EmotionLlamaClient.py` (launches on port `7889`)

Prerequisites
- Docker installed with NVIDIA GPU support (NVIDIA Container Toolkit) and a compatible NVIDIA driver for CUDA 11.7.
- A GPU with sufficient memory. For the bundled 7B model: dockeat least 16 GB GPU RAM recommended; 24 GB+ preferred for comfortable inference.

Model files required (place under repo before running Docker or mount at runtime)
- Checkpoints (LLM): `checkpoints/Llama-2-7b-chat-hf/`
  - Required files (example set present in this repo): `config.json`, `generation_config.json`, `pytorch_model.bin.index.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`, `README.md`, `LICENSE.txt`.
  - Note: Large weight files (pytorch model shards) are *not* included in the repo. Download them from the model provider (Hugging Face / Meta) and put them into `checkpoints/Llama-2-7b-chat-hf/` preserving filenames.

- Vision / audio feature extractor: `transformer/chinese-hubert-large/`
  - Files present in this repo: `chinese-hubert-large-fairseq-ckpt.pt`, `config.json`, `preprocessor_config.json`.
  - Keep these files in place (already included) or re-download from the original source if missing.

- Fine-tuned checkpoint (optional): `save_checkpoint/Emoation_LLaMA.pth`
  - If you have a fine-tuned checkpoint, place it under `save_checkpoint/`.

How to obtain LLM weights
- If the model is hosted on Hugging Face and you have access, use the `huggingface-cli` or `git lfs` to download to the `checkpoints` folder. Example:

```bash
# login and clone (replace with correct repo id if different)
huggingface-cli login
git lfs install
git clone https://huggingface.co/<model-repo-id> checkpoints/Llama-2-7b-chat-hf
```

- If you have no access rights, then you need to copy the `/checkpoint` folder externally

Docker build and run (recommended: mount local model folders)

```bash
# build image
docker build -t emotion-llama:latest .

# run with GPU access and mount model folders (adjust paths)
docker run --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -v $(pwd)/transformer:/app/transformer:ro \
  -v $(pwd)/save_checkpoint:/app/save_checkpoint:ro \
  -p 7860:7860 \
  -e PORT=7860 \
  --rm \
  emotion-llama:latest
```

Notes
- The repository contains two ways to start a Gradio app:
  - `python app.py` — main demo (uses port `7860` by default).
  - `python app_EmotionLlamaClient.py` — alternate client demo (launches on `7889`).
- Dockerfile exposes port `7860` and runs `app.py` by default.
- If you prefer not to bake large weights into the image, mount the `checkpoints` and `save_checkpoint` directories as volumes (see `docker run` above).
- If you encounter CUDA/driver issues, ensure your host driver supports CUDA 11.7 and `nvidia-smi` reports a compatible driver.

Troubleshooting
- If the server returns errors when loading the model, check file permissions and that the model weight shards exist in `checkpoints/Llama-2-7b-chat-hf/`.
- If you see out-of-memory errors, lower `--max_new_tokens` or use a larger GPU, or consider using model quantization (outside this quickstart).

Want me to add
- a `docker-compose.yml` example that mounts models and exposes ports, or
- a small `download_models.sh` helper that uses `huggingface-cli` to fetch known files.

Manual copy from local drives (when Hugging Face access is restricted)
- Sometimes you must copy model files manually (for example, from a corporate drive, external disk, or another machine) because you don't have HF permissions. Follow these steps:

- 1) Identify required files
  - Minimal required files for `checkpoints/Llama-2-7b-chat-hf/`:
    - `config.json`, `generation_config.json`, `pytorch_model.bin.index.json`, tokenizer files (`tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`), `special_tokens_map.json`, and the model shard files referenced by `pytorch_model.bin.index.json` (these are usually large `.bin` or `.safetensors` files).

- 2) Copy files to repository location
  - Windows (PowerShell):

```powershell
# copy from external drive (adjust paths)
Copy-Item -Path "E:\models\Llama-2-7b-chat-hf\*" -Destination "C:\path\to\Emotion-LLaMA\checkpoints\Llama-2-7b-chat-hf\" -Recurse
```

  - Linux / macOS:

```bash
# copy from mounted drive
cp -r /mnt/external/models/Llama-2-7b-chat-hf/* /home/user/Emotion-LLaMA/checkpoints/Llama-2-7b-chat-hf/
```

- 3) Preserve filenames and directory layout
  - Make sure the `pytorch_model.bin.index.json` references match the shard filenames you copied. Do not rename shards.

- 4) Verify integrity (optional but recommended)
  - If you have SHA256 checksums, verify them. Example (Linux):

```bash
sha256sum checkpoints/Llama-2-7b-chat-hf/*.bin
```

  - On Windows (PowerShell):

```powershell
Get-FileHash -Algorithm SHA256 .\checkpoints\Llama-2-7b-chat-hf\*.bin
```

- 5) Adjust permissions if needed
  - Linux: `chmod -R a+r checkpoints/Llama-2-7b-chat-hf/`
  - Windows: ensure your user account has read access to the files.

- 6) Start the server (same as above)
  - Build/run the Docker image and mount the local `checkpoints` directory, or run locally with `python app.py`.

- Troubleshooting notes for manual copy
  - If the server complains about missing shards, open `pytorch_model.bin.index.json` and check the `weight_map` keys — these list the shard filenames required.
  - If you copied only tokenizer files but not weights, the model will fail to load; ensure both tokenizer and weight shards are present.
