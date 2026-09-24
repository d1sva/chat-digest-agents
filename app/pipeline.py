"""Оркестратор: порядок работы агентов и цикл доработки по замечаниям критика.

Раунд:
  1. суммаризатор и аналитик работают параллельно (asyncio.gather);
  2. критик проверяет их результат;
  3. если критик не одобрил и лимит доработок не исчерпан, оба агента
     переделывают работу с учётом замечаний.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, TypeVar

from app.agents import (
    ChatMessage,
    CriticAgent,
    ExtractorAgent,
    Review,
    SummarizerAgent,
    Task,
    format_transcript,
)
from app.llm import LLMClient

T = TypeVar("T")


@dataclass
class TraceStep:
    agent: str
    round: int
    duration_ms: int


@dataclass
class DigestResult:
    summary: str
    tasks: list[Task]
    decisions: list[str]
    review: Review
    rounds: int
    trace: list[TraceStep] = field(default_factory=list)


class DigestPipeline:
    def __init__(self, llm: LLMClient, max_review_rounds: int = 1):
        if max_review_rounds < 0:
            raise ValueError("max_review_rounds не может быть отрицательным")
        self.summarizer = SummarizerAgent(llm)
        self.extractor = ExtractorAgent(llm)
        self.critic = CriticAgent(llm)
        self.max_review_rounds = max_review_rounds

    async def run(self, messages: list[ChatMessage]) -> DigestResult:
        transcript = format_transcript(messages)
        trace: list[TraceStep] = []
        feedback: list[str] | None = None
        round_no = 0

        while True:
            round_no += 1
            summary, extraction = await asyncio.gather(
                self._timed(self.summarizer.name, round_no, trace,
                            self.summarizer.run(transcript, feedback)),
                self._timed(self.extractor.name, round_no, trace,
                            self.extractor.run(transcript, feedback)),
            )
            review = await self._timed(
                self.critic.name, round_no, trace,
                self.critic.run(transcript, summary, extraction),
            )
            if review.approved or round_no > self.max_review_rounds:
                break
            feedback = review.issues

        return DigestResult(
            summary=summary,
            tasks=extraction.tasks,
            decisions=extraction.decisions,
            review=review,
            rounds=round_no,
            trace=trace,
        )

    @staticmethod
    async def _timed(agent: str, round_no: int, trace: list[TraceStep],
                     coro: Awaitable[T]) -> T:
        start = time.perf_counter()
        result = await coro
        trace.append(TraceStep(agent, round_no, int((time.perf_counter() - start) * 1000)))
        return result
