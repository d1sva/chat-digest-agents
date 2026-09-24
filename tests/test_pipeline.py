import asyncio

import pytest

from app.agents import AgentOutputError, ChatMessage, extract_json
from app.pipeline import DigestPipeline
from tests.fakes import ScriptedLLM

MESSAGES = [
    ChatMessage("Аня", "Сделаю презентацию до пятницы"),
    ChatMessage("Боря", "Договорились, защищаемся в понедельник"),
]
EXTRACTION = (
    '{"tasks": [{"text": "Сделать презентацию", "assignee": "Аня", "deadline": "пятница"}],'
    ' "decisions": ["Защита в понедельник"]}'
)
APPROVED = '{"approved": true, "issues": []}'
REJECTED = '{"approved": false, "issues": ["Не указан срок защиты"]}'


def run(pipeline: DigestPipeline):
    return asyncio.run(pipeline.run(MESSAGES))


def test_approved_in_first_round():
    llm = ScriptedLLM({"summarizer": ["Итог"], "extractor": [EXTRACTION], "critic": [APPROVED]})
    result = run(DigestPipeline(llm))

    assert result.rounds == 1
    assert result.summary == "Итог"
    assert result.tasks[0].assignee == "Аня"
    assert result.decisions == ["Защита в понедельник"]
    assert result.review.approved
    assert sorted(step.agent for step in result.trace) == ["critic", "extractor", "summarizer"]


def test_revision_uses_critic_feedback():
    llm = ScriptedLLM({
        "summarizer": ["Черновик", "Исправлено"],
        "extractor": [EXTRACTION],
        "critic": [REJECTED, APPROVED],
    })
    result = run(DigestPipeline(llm, max_review_rounds=1))

    assert result.rounds == 2
    assert result.summary == "Исправлено"
    second_summarizer_prompt = [u for role, u in llm.calls if role == "summarizer"][1]
    assert "Не указан срок защиты" in second_summarizer_prompt


def test_stops_after_max_rounds():
    llm = ScriptedLLM({"summarizer": ["Итог"], "extractor": [EXTRACTION], "critic": [REJECTED]})
    result = run(DigestPipeline(llm, max_review_rounds=2))

    assert result.rounds == 3
    assert not result.review.approved
    assert len([c for c in llm.calls if c[0] == "critic"]) == 3


def test_zero_review_rounds_means_single_pass():
    llm = ScriptedLLM({"summarizer": ["Итог"], "extractor": [EXTRACTION], "critic": [REJECTED]})
    result = run(DigestPipeline(llm, max_review_rounds=0))
    assert result.rounds == 1


def test_null_fields_are_normalized():
    extraction = '{"tasks": [{"text": "Купить торт", "assignee": "null", "deadline": null}], "decisions": []}'
    llm = ScriptedLLM({"summarizer": ["Итог"], "extractor": [extraction], "critic": [APPROVED]})
    task = run(DigestPipeline(llm)).tasks[0]
    assert task.assignee is None and task.deadline is None


def test_invalid_extractor_output_raises():
    llm = ScriptedLLM({"summarizer": ["Итог"], "extractor": ["не JSON"], "critic": [APPROVED]})
    with pytest.raises(AgentOutputError):
        run(DigestPipeline(llm))


def test_extract_json_handles_code_fences():
    assert extract_json('Вот ответ:\n```json\n{"approved": true}\n```') == {"approved": True}


def test_extract_json_rejects_garbage():
    with pytest.raises(AgentOutputError):
        extract_json("{сломанный")
