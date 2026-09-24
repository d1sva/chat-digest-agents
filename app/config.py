"""Настройки сервиса из переменных окружения (.env)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str  # "fake" — демо без LLM, "openai" — любой OpenAI-совместимый API
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    max_review_rounds: int


def get_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "fake").strip().lower(),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b").strip(),
        max_review_rounds=int(os.getenv("MAX_REVIEW_ROUNDS", "1")),
    )
