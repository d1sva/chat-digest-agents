"""Агенты: у каждого своя роль, промпт и формат ответа."""

import json
import re
from dataclasses import asdict, dataclass, field

from app.llm import LLMClient


class AgentOutputError(Exception):
    """Агент вернул ответ, который не удалось разобрать."""


SUMMARIZER_PROMPT = (
    "Ты — агент-суммаризатор. Тебе дают переписку группового чата. "
    "Напиши краткое содержание обсуждения на русском языке: 3–6 предложений, "
    "только факты из переписки, без выдумок. Верни только текст."
)

EXTRACTOR_PROMPT = (
    "Ты — агент-аналитик. Найди в переписке группового чата задачи и принятые решения. "
    "Задача — это конкретное действие, которое кто-то должен сделать. "
    "Верни только JSON без пояснений в формате: "
    '{"tasks": [{"text": "что сделать", "assignee": "имя или null", '
    '"deadline": "срок или null"}], "decisions": ["решение"]}'
)

CRITIC_PROMPT = (
    "Ты — агент-критик. Тебе дают переписку группового чата и результат работы "
    "других агентов: краткое содержание, задачи и решения. Проверь, не упущены ли "
    "важные задачи или решения и нет ли в результате того, чего не было в переписке. "
    "Верни только JSON без пояснений в формате: "
    '{"approved": true или false, "issues": ["замечание"]}'
)


@dataclass
class ChatMessage:
    author: str
    text: str


@dataclass
class Task:
    text: str
    assignee: str | None = None
    deadline: str | None = None


@dataclass
class Extraction:
    tasks: list[Task] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)


@dataclass
class Review:
    approved: bool
    issues: list[str] = field(default_factory=list)


def format_transcript(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.author}: {m.text}" for m in messages)


def extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели (модели любят оборачивать его в ```json)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise AgentOutputError("В ответе нет JSON-объекта")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as e:
        raise AgentOutputError("Некорректный JSON в ответе") from e
    if not isinstance(data, dict):
        raise AgentOutputError("Ожидался JSON-объект")
    return data


def _with_feedback(user: str, feedback: list[str] | None) -> str:
    if not feedback:
        return user
    notes = "\n".join(f"- {item}" for item in feedback)
    return f"{user}\n\nЗамечания критика к прошлой версии, учти их:\n{notes}"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"null", "none"} else None


class SummarizerAgent:
    name = "summarizer"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(self, transcript: str, feedback: list[str] | None = None) -> str:
        user = _with_feedback(f"Переписка:\n{transcript}", feedback)
        answer = (await self.llm.complete(SUMMARIZER_PROMPT, user)).strip()
        if not answer:
            raise AgentOutputError("Суммаризатор вернул пустой ответ")
        return answer


class ExtractorAgent:
    name = "extractor"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(self, transcript: str, feedback: list[str] | None = None) -> Extraction:
        user = _with_feedback(f"Переписка:\n{transcript}", feedback)
        data = extract_json(await self.llm.complete(EXTRACTOR_PROMPT, user))

        tasks = []
        for item in data.get("tasks") or []:
            if isinstance(item, dict) and str(item.get("text") or "").strip():
                tasks.append(
                    Task(
                        text=str(item["text"]).strip(),
                        assignee=_optional_str(item.get("assignee")),
                        deadline=_optional_str(item.get("deadline")),
                    )
                )
        decisions = [str(d).strip() for d in data.get("decisions") or [] if str(d).strip()]
        return Extraction(tasks=tasks, decisions=decisions)


class CriticAgent:
    name = "critic"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def run(self, transcript: str, summary: str, extraction: Extraction) -> Review:
        result = json.dumps(
            {
                "summary": summary,
                "tasks": [asdict(t) for t in extraction.tasks],
                "decisions": extraction.decisions,
            },
            ensure_ascii=False,
            indent=2,
        )
        user = f"Переписка:\n{transcript}\n\nРезультат агентов:\n{result}"
        data = extract_json(await self.llm.complete(CRITIC_PROMPT, user))
        issues = [str(i).strip() for i in data.get("issues") or [] if str(i).strip()]
        return Review(approved=data.get("approved") is True, issues=issues)
