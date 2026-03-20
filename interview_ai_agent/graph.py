from __future__ import annotations

import os
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from interview_ai_agent.services.audio_content import run_audio_content_analysis
from interview_ai_agent.services.scoring import score_candidate
from interview_ai_agent.services.validation import validate_scorecard_format
from interview_ai_agent.services.video_analysis import analyze_video_signal
from interview_ai_agent.state import InterviewAgentState


def _step_max_attempts(state: InterviewAgentState) -> int:
    return max(1, int(state.get("step_max_attempts", 3)))


def _step_retry_delay_sec(state: InterviewAgentState) -> float:
    return max(0.0, float(state.get("step_retry_delay_sec", 1.0)))


def _run_step_with_retry(
    step_name: str,
    state: InterviewAgentState,
    fn,
) -> InterviewAgentState:
    last_error: Exception | None = None
    attempts = _step_max_attempts(state)
    delay = _step_retry_delay_sec(state)

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(delay)

    return {"error": f"{step_name} failed after {attempts} attempts: {last_error}"}


def _prepare(state: InterviewAgentState) -> InterviewAgentState:
    def _inner() -> InterviewAgentState:
        video_path = state["video_path"]
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}
        return {}

    return _run_step_with_retry("prepare", state, _inner)


def _video_agent(state: InterviewAgentState) -> InterviewAgentState:
    def _inner() -> InterviewAgentState:
        return {"video_analysis": analyze_video_signal(state["video_path"])}

    return _run_step_with_retry("video_agent", state, _inner)


def _audio_content_agent(state: InterviewAgentState) -> InterviewAgentState:
    def _inner() -> InterviewAgentState:
        result = run_audio_content_analysis(
            video_path=state["video_path"],
            cv_text=state["cv_text"],
            job_description=state["job_description"],
        )
        return {"audio_content_analysis": result}

    return _run_step_with_retry("audio_content_agent", state, _inner)


def _scoring_agent(state: InterviewAgentState) -> InterviewAgentState:
    if state.get("error"):
        return {}

    def _inner() -> InterviewAgentState:
        scorecard = score_candidate(
            video_analysis=state.get("video_analysis", {}),
            audio_content_analysis=state.get("audio_content_analysis", {}),
            job_description=state.get("job_description", ""),
            cv_text=state.get("cv_text", ""),
            feedback=state.get("validation_feedback", ""),
        )
        return {"scorecard": scorecard}

    return _run_step_with_retry("scoring_agent", state, _inner)


def _checking_agent(state: InterviewAgentState) -> InterviewAgentState:
    if state.get("error"):
        return {"is_valid": False, "validation_feedback": state["error"]}

    def _inner() -> InterviewAgentState:
        valid, feedback = validate_scorecard_format(state.get("scorecard", {}))
        retries = int(state.get("retry_count", 0))

        updates: InterviewAgentState = {
            "is_valid": valid,
            "validation_feedback": feedback,
        }

        if valid:
            updates["validated_output"] = state.get("scorecard", {})
        else:
            updates["retry_count"] = retries + 1

        return updates

    return _run_step_with_retry("checking_agent", state, _inner)


def _route_after_check(state: InterviewAgentState) -> str:
    if state.get("is_valid"):
        return "done"

    if int(state.get("retry_count", 0)) >= int(state.get("max_retries", 2)):
        return "maxed"

    return "retry"


def build_graph():
    builder = StateGraph(InterviewAgentState)

    builder.add_node("prepare", _prepare)
    builder.add_node("video_agent", _video_agent)
    builder.add_node("audio_content_agent", _audio_content_agent)
    builder.add_node("scoring_agent", _scoring_agent)
    builder.add_node("checking_agent", _checking_agent)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "video_agent")
    builder.add_edge("prepare", "audio_content_agent")

    # Wait for both parallel branches before scoring.
    builder.add_edge("video_agent", "scoring_agent")
    builder.add_edge("audio_content_agent", "scoring_agent")

    builder.add_edge("scoring_agent", "checking_agent")
    builder.add_conditional_edges(
        "checking_agent",
        _route_after_check,
        {
            "done": END,
            "retry": "scoring_agent",
            "maxed": END,
        },
    )

    return builder.compile()


def run_interview_evaluation(
    video_path: str,
    job_description: str,
    cv_text: str,
    max_retries: int = 2,
    step_max_attempts: int = 3,
    step_retry_delay_sec: float = 1.0,
) -> dict[str, Any]:
    graph = build_graph()
    initial_state: InterviewAgentState = {
        "video_path": video_path,
        "job_description": job_description,
        "cv_text": cv_text,
        "retry_count": 0,
        "max_retries": max_retries,
        "step_max_attempts": step_max_attempts,
        "step_retry_delay_sec": step_retry_delay_sec,
        "is_valid": False,
        "validation_feedback": "",
    }
    return graph.invoke(initial_state)
