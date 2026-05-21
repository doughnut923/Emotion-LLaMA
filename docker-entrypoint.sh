#!/bin/sh
set -e

# Enable hf-transfer for faster HF Hub downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Default paths (can be overridden via env vars)
: ${MODEL_PATH:=/app/checkpoints/Llama-2-7b-chat-hf}
: ${HUBERT_PATH:=/app/checkpoints/transformer/chinese-hubert-large}
: ${CKPT_PATH:=/app/checkpoints/save_checkpoint/Emoation_LLaMA.pth}

missing=0
echo "Checking required model files..."
if [ ! -d "$MODEL_PATH" ]; then
  echo "LLaMA model not found at $MODEL_PATH"
  # Try runtime download from Hugging Face if HF_TOKEN and HF_MODEL_REPO provided
  HF_MODEL_REPO=${HF_MODEL_REPO:-doughnut23/emollama-models}
  HF_REPO_TYPE=${HF_REPO_TYPE:-model}
  if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN provided; attempting to download $HF_MODEL_REPO (type=$HF_REPO_TYPE) into /app/checkpoints"
    # Ensure parent checkpoints dir exists
    mkdir -p /app/checkpoints
    if [ -w /app/checkpoints ]; then
      echo "Downloading into /app/checkpoints (this may take a while)..."
      python - <<'PY' || true
from huggingface_hub import snapshot_download
import os, sys
repo = os.environ.get('HF_MODEL_REPO', 'doughnut23/emollama-models')
repo_type = os.environ.get('HF_REPO_TYPE', 'model')
token = os.environ.get('HF_TOKEN')
try:
    snapshot_download(repo_id=repo, repo_type=repo_type, local_dir='/app/checkpoints', use_auth_token=token)
    print('Snapshot download finished')
except Exception as e:
    print('Snapshot download failed:', e)
    sys.exit(1)
PY
      if [ -d "$MODEL_PATH" ]; then
        echo "Downloaded LLaMA model to $MODEL_PATH"
      else
        echo "Download completed but model path $MODEL_PATH still missing."
        missing=1
      fi
    else
      echo "Cannot write to /app/checkpoints. If you mounted ./checkpoints as read-only, remove the mount or make it writable to allow runtime download."
      missing=1
    fi
  else
    echo "No HF_TOKEN provided; cannot attempt download of $HF_DATASET_REPO."
    missing=1
  fi
else
  echo "Found LLaMA model at $MODEL_PATH"
fi

if [ ! -d "$HUBERT_PATH" ]; then
  echo "WARNING: HuBERT model not found at $HUBERT_PATH (some features may be limited)"
else
  echo "Found HuBERT model at $HUBERT_PATH"
fi

if [ ! -f "$CKPT_PATH" ]; then
  echo "WARNING: Emotion-LLaMA checkpoint not found at $CKPT_PATH (demo may have degraded behavior)"
else
  echo "Found Emotion-LLaMA checkpoint at $CKPT_PATH"
fi

if [ "$missing" -eq 1 ]; then
  echo "\nPlease mount your model files into /app/checkpoints when running the container. Example:\n"
  echo "  docker run -v /host/path/checkpoints:/app/checkpoints:ro -p 7860:7860 emotion-llama:latest"
  echo "Exiting."
  exit 1
fi

exec "$@"
