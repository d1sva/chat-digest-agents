"""Клиенты языковых моделей.

Агенты зависят только от протокола LLMClient, поэтому модель можно
заменить (Ollama, OpenRouter, демо-заглушка, заглушка в тестах),
не меняя код агентов.
"""

from typing import Protocol

import httpx


class LLMError(Exception):
    """Ошибка обращения к языковой модели."""


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...

    async def aclose(self) -> None: ...


class OpenAICompatibleClient:
    """Клиент для любого API в формате OpenAI /chat/completions."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
        )

    async def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        try:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise LLMError(f"LLM вернула статус {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM недоступна: {e.__class__.__name__}") from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError("Неожиданный формат ответа LLM") from e

    async def aclose(self) -> None:
        await self._client.aclose()
