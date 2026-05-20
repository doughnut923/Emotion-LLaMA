FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

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
COPY configs/ ./configs/

# Add entrypoint that checks for mounted checkpoints at runtime
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "app.py"]