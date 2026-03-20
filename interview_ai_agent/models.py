from pydantic import BaseModel, Field


class CriteriaScores(BaseModel):
    communication: float = Field(ge=0, le=10)
    domain_fit: float = Field(ge=0, le=10)
    role_alignment: float = Field(ge=0, le=10)
    confidence_and_presence: float = Field(ge=0, le=10)
    overall: float = Field(ge=0, le=10)


class Scorecard(BaseModel):
    candidate_summary: str
    strengths: list[str]
    weaknesses: list[str]
    criteria_scores: CriteriaScores
    hiring_recommendation: str
    confidence: float = Field(ge=0, le=1)
    rationale: str


REQUIRED_RECOMMENDATIONS = {"strong_yes", "yes", "lean_no", "no"}
