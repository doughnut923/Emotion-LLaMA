import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from interview_ai_agent.graph import run_interview_evaluation


def _read_text_arg(raw: str) -> str:
    p = Path(raw)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return raw


def main() -> None:
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env")

    parser = argparse.ArgumentParser(description="Run interview evaluation LangGraph pipeline")
    parser.add_argument("--video", required=True, help="Path to input interview .mp4")
    parser.add_argument("--jd", required=True, help="Job description text OR a file path")
    parser.add_argument("--cv", required=True, help="CV text OR a file path")
    parser.add_argument("--max-retries", type=int, default=2, help="Max validation retry attempts")
    parser.add_argument("--step-max-attempts", type=int, default=3, help="Max attempts per step on unexpected errors")
    parser.add_argument("--step-retry-delay-sec", type=float, default=1.0, help="Delay in seconds between step retries")
    args = parser.parse_args()

    result = run_interview_evaluation(
        video_path=args.video,
        job_description=_read_text_arg(args.jd),
        cv_text=_read_text_arg(args.cv),
        max_retries=args.max_retries,
        step_max_attempts=args.step_max_attempts,
        step_retry_delay_sec=args.step_retry_delay_sec,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
