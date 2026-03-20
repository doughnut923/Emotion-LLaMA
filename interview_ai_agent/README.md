# Interview AI Agent (LangGraph)

This folder contains a dedicated pipeline for interview video evaluation.

## Input
- `mp4` video path
- Job Description (`string` or text file path)
- CV (`string` or text file path)

## Workflow
1. **Video Agent** (parallel branch): calls local Emotion-LLaMA API and extracts tone, engagement, body language.
2. **Audio/Content Agent** (parallel branch):
   - extracts audio from video
  - transcribes audio (OpenAI Whisper)
  - evaluates transcript against CV and JD using an LLM structured output
3. **Scoring Agent**: combines both branches into a structured scorecard and generates candidate summary via LLM.
4. **Checking Agent**: validates output format. If invalid, routes back to scoring until valid or retries exceeded.

## Output Shape
- `candidate_summary`
- `strengths[]`
- `weaknesses[]`
- `criteria_scores`
  - `communication`
  - `domain_fit`
  - `role_alignment`
  - `confidence_and_presence`
  - `overall`
- `hiring_recommendation` (`strong_yes|yes|lean_no|no`)
- `confidence` (`0..1`)
- `rationale`

## Install
```bash
pip install -r interview_ai_agent/requirements.txt
```

## Run
```bash
python -m interview_ai_agent.run_pipeline \
  --video path/to/interview.mp4 \
  --jd "We need a backend engineer with Python and API design experience." \
  --cv "5 years Python, FastAPI, Docker, microservices..." \
  --step-max-attempts 3 \
  --step-retry-delay-sec 1.0
```

You can also pass file paths for `--jd` and `--cv`.

## Environment Variables
- `EMOTION_LLAMA_API_URL` (optional): defaults to `http://127.0.0.1:7889/api/predict/`
- `OPEN_ROUTER_KEY` or `OPENROUTER_API_KEY` (required for LLM-based analysis via OpenRouter)
- `OPENROUTER_BASE_URL` (optional): defaults to `https://openrouter.ai/api/v1`
- `INTERVIEW_AGENT_LLM_MODEL` (optional): defaults to `openai/gpt-oss-120b:free`
- `OPENROUTER_APP_NAME` (optional): defaults to `interview-ai-agent`
- `OPENROUTER_APP_URL` (optional): defaults to `http://localhost`

The video branch expects the Emotion-LLaMA API contract that accepts:
```json
{"data": ["/path/to/video.mp4", "[emotion] ..."]}
```
If you are running the Gradio UI-only endpoint on `7860`, start the path-based API service (for example `app_EmotionLlamaClient.py`) and point `EMOTION_LLAMA_API_URL` to it.

## Notes
- Video analysis is now powered by local Emotion-LLaMA API calls.
- If LLM calls fail, the pipeline falls back to deterministic scoring-safe values so graph execution can continue.
- Each step (prepare, video, audio-content, scoring, checking) automatically retries on unexpected exceptions.
