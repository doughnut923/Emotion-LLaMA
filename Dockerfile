FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Build args for HF token and model repo
ARG HF_TOKEN
ARG HF_MODEL_REPO=doughnut23/emollama-models
ARG HF_REPO_TYPE=model

# Copy and install only dependencies first to leverage layer cache
COPY requirements.txt /app/

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files (checkpoints and large artifacts are excluded by .dockerignore)
# Keep copy scope explicit to avoid accidental large copies
COPY app.py app_EmotionLlamaClient.py infer_api.py ./
COPY minigpt4/ ./minigpt4/
COPY eval_configs/ ./eval_configs/

# Add entrypoint that checks for mounted checkpoints at runtime
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Build-time download from Hugging Face if HF_TOKEN is provided.
# Pass via: docker-compose build --build-arg HF_TOKEN=... or docker build --build-arg HF_TOKEN=...
RUN test -n "$HF_TOKEN" && python - <<'PY' || echo "No HF_TOKEN provided; skipping build-time download"
import os,sys
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN')
repo = os.environ.get('HF_MODEL_REPO','doughnut23/emollama-models')
repo_type = os.environ.get('HF_REPO_TYPE','model')
if token:
    try:
        print('snapshot_download:', repo, '(type=' + repo_type + ')')
        snapshot_download(repo_id=repo, repo_type=repo_type, local_dir='/app/checkpoints', use_auth_token=token)
        print('Build-time download completed')
    except Exception as e:
        print('Build-time download failed:', e)
        sys.exit(0)
else:
    print('No HF_TOKEN; skipping build-time download')
PY

EXPOSE 7860

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "app.py"]