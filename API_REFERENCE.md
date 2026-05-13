# Emotion-LLaMA — API Reference

This document describes the HTTP API exposed by the Emotion-LLaMA demo (Gradio). The primary endpoint is the Gradio `predict` endpoint used by clients in this repo.

## Base URL
- Default local server: `http://127.0.0.1:7889`

## POST /api/predict/
- Description: Gradio `predict` endpoint serving the model. It accepts a JSON body with a `data` array and returns JSON with a `data` array where the first element is the model text output.

- Request
  - Content-Type: `application/json`
  - Body schema: `{ "data": [ <image_or_video_path_or_base64>, "<prompt>" ] }`
  - Notes:
    - The first element may be:
      - a video or image file path accessible to the server (e.g. `"/path/to/video.mp4"`), or
      - a Base64-encoded PNG/JPEG string (recommended when sending files from a remote client).
    - Many client utilities in this repo try both strategies (path first, then base64). See `interview_ai_agent/services/video_analysis.py` for example behavior.

- Example request body (JSON):

```json
{
  "data": ["iVBORw0KGgoAAAANSUhEUgAA...base64-image...", "Analyze this video and return the emotion of the person."]
}
```

- Curl example:

```bash
curl -X POST "http://127.0.0.1:7889/api/predict/" \
  -H "Content-Type: application/json" \
  -d '{"data": ["<BASE64_IMAGE_OR_PATH>", "What emotions does this convey?"]}'
```

- Python example (requests):

```python
import requests, json

url = "http://127.0.0.1:7889/api/predict/"
payload = {"data": [base64_image_or_path, "Analyze this frame for emotions."]}
resp = requests.post(url, json=payload, timeout=240)
resp.raise_for_status()
result = resp.json()
print(result.get("data", [""])[0])
```

- Response (typical):

```json
{
  "data": ["<model textual response here>"],
  "is_generating": false
}
```

## Notes and implementation pointers
- The server in this repository launches a Gradio interface which exposes the `/api/predict/` endpoint. See the Gradio app that starts the server:

- `app_EmotionLlamaClient.py` (launches Gradio on port 7889 by default): [app_EmotionLlamaClient.py](app_EmotionLlamaClient.py#L1-L260)
- Helper client that posts to the endpoint: [infer_api.py](infer_api.py#L1-L200)
- Higher-level consumer that tries both file-path and base64 strategies: [interview_ai_agent/services/video_analysis.py](interview_ai_agent/services/video_analysis.py#L1-L200)

- To run the server locally:

```bash
python app_EmotionLlamaClient.py
```

- Environment variable used by consumer code to point to the API: `EMOTION_LLAMA_API_URL` (defaults to `http://127.0.0.1:7889/api/predict/`). See `interview_ai_agent/services/video_analysis.py`.

## Quick troubleshooting
- If you send a file path, the server process must have access to that path.
- If you get JSON decode errors, try sending the Base64-encoded PNG of the first frame instead of a path.

---

If you want, I can:
- add more endpoints or examples (e.g., multipart/form-data upload),
- include a minimal client module (requests wrapper), or
- add a short curl + base64 conversion helper snippet.
