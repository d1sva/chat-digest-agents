"""Схемы запросов и ответов API (валидация через Pydantic)."""

from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    author: str = Field(min_length=1, max_length=100, examples=["Аня"])
    text: str = Field(min_length=1, max_length=4000, examples=["Сделаю презентацию до пятницы"])


class DigestRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1, max_length=500)


class TaskOut(BaseModel):
    text: str
    assignee: str | None
    deadline: str | None


class ReviewOut(BaseModel):
    approved: bool
    issues: list[str]


class TraceStepOut(BaseModel):
    agent: str
    round: int
    duration_ms: int


class DigestResponse(BaseModel):
    summary: str
    tasks: list[TaskOut]
    decisions: list[str]
    review: ReviewOut
    rounds: int = Field(description="Сколько раундов понадобилось до одобрения критиком")
    trace: list[TraceStepOut] = Field(description="Какой агент в каком раунде сколько работал")
