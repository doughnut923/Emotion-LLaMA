from __future__ import annotations

from pydantic import ValidationError

from interview_ai_agent.models import REQUIRED_RECOMMENDATIONS, Scorecard


def validate_scorecard_format(scorecard: dict) -> tuple[bool, str]:
    if not isinstance(scorecard, dict):
        return False, "Scorecard must be a JSON object"

    try:
        parsed = Scorecard.model_validate(scorecard)
    except ValidationError as exc:
        return False, f"Schema validation failed: {exc}"

    if parsed.hiring_recommendation not in REQUIRED_RECOMMENDATIONS:
        return False, (
            "hiring_recommendation must be one of "
            f"{sorted(REQUIRED_RECOMMENDATIONS)}"
        )

    if len(parsed.strengths) == 0 or len(parsed.weaknesses) == 0:
        return False, "Both strengths and weaknesses must contain at least one item"

    return True, "ok"
