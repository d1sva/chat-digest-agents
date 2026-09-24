"""Заглушка LLM для тестов: отвечает заранее заданными ответами по ролям."""

from app.agents import CRITIC_PROMPT, EXTRACTOR_PROMPT, SUMMARIZER_PROMPT

ROLE_BY_PROMPT = {
    SUMMARIZER_PROMPT: "summarizer",
    EXTRACTOR_PROMPT: "extractor",
    CRITIC_PROMPT: "critic",
}


class ScriptedLLM:
    def __init__(self, responses: dict[str, list[str]]):
        # для каждой роли — очередь ответов; последний ответ повторяется
        self.responses = {role: list(items) for role, items in responses.items()}
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        role = ROLE_BY_PROMPT[system]
        self.calls.append((role, user))
        queue = self.responses[role]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    async def aclose(self) -> None:
        return None


class FailingLLM:
    async def complete(self, system: str, user: str) -> str:
        from app.llm import LLMError
        raise LLMError("модель недоступна")

    async def aclose(self) -> None:
        return None
