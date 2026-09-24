"""Демо-режим без настоящей LLM (LLM_PROVIDER=fake).

Нужен, чтобы запустить и посмотреть API без модели и ключей.
Отвечает по простым правилам: задачи ищет по словам «сделаю», «возьму» и т.п.,
решения по словам «решили», «договорились».
"""

import json
import re

from app.agents import CRITIC_PROMPT, EXTRACTOR_PROMPT, SUMMARIZER_PROMPT

TASK_WORDS = ("сделаю", "возьму", "беру", "подготовлю", "напишу", "займусь")
DECISION_WORDS = ("решили", "договорились", "решено", "итого")


def _parse_lines(user: str) -> list[tuple[str, str]]:
    body = user.split("Переписка:\n", 1)[-1].split("\n\n", 1)[0]
    lines = []
    for line in body.splitlines():
        if ": " in line:
            author, text = line.split(": ", 1)
            lines.append((author.strip(), text.strip()))
    return lines


class DemoLLM:
    async def complete(self, system: str, user: str) -> str:
        lines = _parse_lines(user)

        if system == SUMMARIZER_PROMPT:
            authors = sorted({a for a, _ in lines})
            return (
                f"[Демо-режим] В обсуждении {len(lines)} сообщений "
                f"от участников: {', '.join(authors)}."
            )

        if system == EXTRACTOR_PROMPT:
            tasks, decisions = [], []
            for author, text in lines:
                low = text.lower()
                if any(w in low for w in TASK_WORDS):
                    deadline = re.search(r"до [\wа-яё.]+", low)
                    tasks.append({
                        "text": text,
                        "assignee": author,
                        "deadline": deadline.group(0) if deadline else None,
                    })
                if any(w in low for w in DECISION_WORDS):
                    decisions.append(text)
            return json.dumps({"tasks": tasks, "decisions": decisions}, ensure_ascii=False)

        if system == CRITIC_PROMPT:
            return json.dumps({"approved": True, "issues": []})

        return ""

    async def aclose(self) -> None:
        return None
