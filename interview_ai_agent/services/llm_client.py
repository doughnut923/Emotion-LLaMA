from __future__ import annotations

import os
from typing import Any, Type

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    api_key = (
        os.getenv("OPEN_ROUTER_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("OPEN_ROUTER_KEY or OPENROUTER_API_KEY is required for LLM-based analysis")

    model = os.getenv("INTERVIEW_AGENT_LLM_MODEL", "openai/gpt-oss-120b:free")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    app_name = os.getenv("OPENROUTER_APP_NAME", "interview-ai-agent")
    app_url = os.getenv("OPENROUTER_APP_URL", "http://localhost")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "HTTP-Referer": app_url,
            "X-Title": app_name,
        },
    )


def invoke_structured(prompt: str, schema: Type[BaseModel], temperature: float = 0.1) -> Any:
    llm = get_llm(temperature=temperature)
    structured_llm = llm.with_structured_output(schema)
    return structured_llm.invoke(prompt)
