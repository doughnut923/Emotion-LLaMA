import os
import re

from moviepy.editor import VideoFileClip
from pydantic import BaseModel, Field

from interview_ai_agent.services.llm_client import invoke_structured


def extract_audio(video_path: str, output_wav: str) -> str:
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("Input video has no audio stream")
    clip.audio.write_audiofile(output_wav, fps=16000, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
    clip.close()
    return output_wav


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z]{3,}", text.lower())]


def _keyword_overlap(reference: str, transcript: str) -> float:
    ref_tokens = set(_tokenize(reference))
    tr_tokens = set(_tokenize(transcript))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & tr_tokens) / len(ref_tokens)


class AudioContentLLMOutput(BaseModel):
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    communication_score: float = Field(ge=0, le=10)
    domain_fit_score: float = Field(ge=0, le=10)
    role_alignment_score: float = Field(ge=0, le=10)
    content_score: float = Field(ge=0, le=10)


def transcribe_audio(audio_path: str) -> str:
    """Use OpenAI Whisper if OPENAI_API_KEY is available, else return placeholder."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Transcript unavailable (set OPENAI_API_KEY to enable Whisper transcription)."

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return result.text


def evaluate_transcript_against_cv_jd(transcript: str, cv_text: str, job_description: str) -> dict:
    cv_overlap = _keyword_overlap(cv_text, transcript)
    jd_overlap = _keyword_overlap(job_description, transcript)

    llm_used = False
    llm_output: AudioContentLLMOutput | None = None
    if "Transcript unavailable" not in transcript:
        prompt = (
            "You are evaluating a candidate interview transcript against CV and job description. "
            "Return structured assessment with concise strengths/weaknesses and numeric scores.\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Candidate CV:\n{cv_text}\n\n"
            f"Transcript:\n{transcript}\n"
        )
        try:
            llm_output = invoke_structured(prompt, AudioContentLLMOutput, temperature=0.1)
            llm_used = True
        except Exception:
            llm_output = None

    if llm_output is not None:
        strengths = llm_output.strengths or ["Candidate provided relevant interview content."]
        weaknesses = llm_output.weaknesses or ["Some competency areas need deeper evidence."]
        content_score = float(llm_output.content_score)
        summary = llm_output.summary
        communication_score = float(llm_output.communication_score)
        domain_fit_score = float(llm_output.domain_fit_score)
        role_alignment_score = float(llm_output.role_alignment_score)
    else:
        strengths = []
        weaknesses = []

        if cv_overlap >= 0.18:
            strengths.append("Candidate discussion aligns with stated CV experience.")
        else:
            weaknesses.append("Limited explicit linkage between answers and CV achievements.")

        if jd_overlap >= 0.16:
            strengths.append("Interview content references responsibilities relevant to the job description.")
        else:
            weaknesses.append("Interview answers show weak coverage of JD-specific priorities.")

        if "Transcript unavailable" in transcript:
            weaknesses.append("Audio transcription unavailable; content evaluation confidence is reduced.")

        content_score = min(10.0, max(0.0, 2.0 + 20.0 * (0.55 * cv_overlap + 0.45 * jd_overlap)))
        communication_score = content_score
        domain_fit_score = min(10.0, max(0.0, 10.0 * cv_overlap))
        role_alignment_score = min(10.0, max(0.0, 10.0 * jd_overlap))
        summary = "Audio/text branch evaluated transcript relevance to CV and JD."

    return {
        "transcript": transcript,
        "content_metrics": {
            "cv_keyword_overlap": round(cv_overlap, 3),
            "jd_keyword_overlap": round(jd_overlap, 3),
            "content_score": round(content_score, 2),
            "communication_score": round(communication_score, 2),
            "domain_fit_score": round(domain_fit_score, 2),
            "role_alignment_score": round(role_alignment_score, 2),
            "llm_used": llm_used,
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "summary": summary,
    }


def run_audio_content_analysis(video_path: str, cv_text: str, job_description: str) -> dict:
    audio_path = os.path.splitext(video_path)[0] + "_temp_agent.wav"
    extract_audio(video_path, audio_path)
    transcript = transcribe_audio(audio_path)
    try:
        os.remove(audio_path)
    except OSError:
        pass
    return evaluate_transcript_against_cv_jd(transcript, cv_text, job_description)
