#!/bin/sh
set -e

# Default paths (can be overridden via env vars)
: ${MODEL_PATH:=/app/checkpoints/Llama-2-7b-chat-hf}
: ${HUBERT_PATH:=/app/checkpoints/transformer/chinese-hubert-large}
: ${CKPT_PATH:=/app/checkpoints/save_checkpoint/Emoation_LLaMA.pth}

missing=0
echo "Checking required model files..."
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: LLaMA model not found at $MODEL_PATH"
  missing=1
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
