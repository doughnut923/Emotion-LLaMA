from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from interview_ai_agent.models import REQUIRED_RECOMMENDATIONS
from interview_ai_agent.services.llm_client import invoke_structured


class CandidateSummaryOutput(BaseModel):
    candidate_summary: str


def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def _llm_candidate_summary(
    video_analysis: dict[str, Any],
    audio_content_analysis: dict[str, Any],
    job_description: str,
    cv_text: str,
    criteria_scores: dict[str, float],
) -> str:
    prompt = (
        "Write a concise candidate summary (3-5 sentences) for a hiring panel. "
        "Use interview visual analysis, transcript analysis, CV and JD fit evidence, and score signals."
        " Avoid bullet points.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate CV:\n{cv_text}\n\n"
        f"Video Analysis:\n{video_analysis}\n\n"
        f"Audio/Content Analysis:\n{audio_content_analysis}\n\n"
        f"Criteria Scores:\n{criteria_scores}\n"
    )
    result = invoke_structured(prompt, CandidateSummaryOutput, temperature=0.2)
    return result.candidate_summary


def score_candidate(
    video_analysis: dict[str, Any],
    audio_content_analysis: dict[str, Any],
    job_description: str,
    cv_text: str,
    feedback: str = "",
) -> dict[str, Any]:
    v = video_analysis.get("visual_metrics", {})
    a = audio_content_analysis.get("content_metrics", {})

    communication = _clamp(0.6 * float(a.get("communication_score", a.get("content_score", 4.5))) + 0.4 * float(v.get("engagement_score", 4.5)))
    domain_fit = _clamp(0.75 * float(a.get("domain_fit_score", 10.0 * float(a.get("cv_keyword_overlap", 0.25)))) + 0.25 * float(a.get("content_score", 4.5)))
    role_alignment = _clamp(0.75 * float(a.get("role_alignment_score", 10.0 * float(a.get("jd_keyword_overlap", 0.25)))) + 0.25 * float(a.get("content_score", 4.5)))
    confidence_and_presence = _clamp(0.6 * float(v.get("presence_score", 4.5)) + 0.4 * float(v.get("engagement_score", 4.5)))
    overall = _clamp((communication + domain_fit + role_alignment + confidence_and_presence) / 4.0)

    strengths = list(dict.fromkeys(audio_content_analysis.get("strengths", []) + [video_analysis.get("body_language", "")]))
    weaknesses = list(dict.fromkeys(audio_content_analysis.get("weaknesses", [])))

    if overall >= 8.0:
        recommendation = "strong_yes"
    elif overall >= 6.5:
        recommendation = "yes"
    elif overall >= 5.0:
        recommendation = "lean_no"
    else:
        recommendation = "no"

    if recommendation not in REQUIRED_RECOMMENDATIONS:
        recommendation = "lean_no"

    rationale = "Combined visual behavior and transcript relevance metrics."
    if feedback:
        rationale += f" Validator feedback applied: {feedback}"

    criteria_scores = {
        "communication": round(communication, 2),
        "domain_fit": round(domain_fit, 2),
        "role_alignment": round(role_alignment, 2),
        "confidence_and_presence": round(confidence_and_presence, 2),
        "overall": round(overall, 2),
    }

    try:
        candidate_summary = _llm_candidate_summary(
            video_analysis=video_analysis,
            audio_content_analysis=audio_content_analysis,
            job_description=job_description,
            cv_text=cv_text,
            criteria_scores=criteria_scores,
        )
    except Exception:
        candidate_summary = (
            "Candidate shows {tone} with engagement around {engagement}. Content relevance to CV/JD "
            "was used for role fit scoring."
        ).format(
            tone=video_analysis.get("tone", "unknown tone"),
            engagement=video_analysis.get("engagement", "unknown"),
        )

    return {
        "candidate_summary": candidate_summary,
        "strengths": [s for s in strengths if s],
        "weaknesses": [w for w in weaknesses if w],
        "criteria_scores": criteria_scores,
        "hiring_recommendation": recommendation,
        "confidence": round(min(1.0, max(0.2, overall / 10.0)), 2),
        "rationale": rationale,
    }
