#!/bin/bash

# Start both Gradio apps
python app.py &
python app_EmotionLlamaClient.py &

# Wait for all background processes
wait
