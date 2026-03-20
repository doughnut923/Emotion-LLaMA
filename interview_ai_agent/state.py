from typing import Any, Optional, TypedDict


class InterviewAgentState(TypedDict, total=False):
    # Inputs
    video_path: str
    job_description: str
    cv_text: str

    # Branch outputs
    video_analysis: dict[str, Any]
    audio_content_analysis: dict[str, Any]

    # Aggregated outputs
    scorecard: dict[str, Any]
    validated_output: dict[str, Any]

    # Validation control
    is_valid: bool
    validation_feedback: str
    retry_count: int
    max_retries: int
    step_max_attempts: int
    step_retry_delay_sec: float

    # Error handling
    error: Optional[str]
