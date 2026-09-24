"""HTTP API сервиса.

Запуск: uvicorn app.main:app --reload
Веб-интерфейс: http://127.0.0.1:8000/
Документация API: http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from app.agents import AgentOutputError, ChatMessage
from app.config import get_settings
from app.demo_llm import DemoLLM
from app.llm import LLMClient, LLMError, OpenAICompatibleClient
from app.pipeline import DigestPipeline
from app.schemas import DigestRequest, DigestResponse

INDEX_HTML = Path(__file__).parent / "static" / "index.html"


def build_llm() -> tuple[LLMClient, int]:
    settings = get_settings()
    if settings.llm_provider == "fake":
        return DemoLLM(), settings.max_review_rounds
    if settings.llm_provider == "openai":
        client = OpenAICompatibleClient(
            settings.llm_base_url, settings.llm_api_key, settings.llm_model
        )
        return client, settings.max_review_rounds
    raise RuntimeError(f"Неизвестный LLM_PROVIDER: {settings.llm_provider}")


def create_app(llm: LLMClient | None = None, max_review_rounds: int = 1) -> FastAPI:
    """Фабрика приложения. В тестах сюда передаётся заглушка вместо LLM."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client, rounds = (llm, max_review_rounds) if llm is not None else build_llm()
        app.state.pipeline = DigestPipeline(client, max_review_rounds=rounds)
        yield
        await client.aclose()

    app = FastAPI(
        title="Chat Digest Agents",
        description="Мультиагентный разбор переписки группового чата: "
                    "краткое содержание, задачи и решения.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(INDEX_HTML)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/digest", response_model=DigestResponse)
    async def digest(body: DigestRequest, request: Request) -> DigestResponse:
        pipeline: DigestPipeline = request.app.state.pipeline
        messages = [ChatMessage(author=m.author, text=m.text) for m in body.messages]
        try:
            result = await pipeline.run(messages)
        except LLMError as e:
            raise HTTPException(status_code=502, detail=f"Ошибка LLM: {e}") from e
        except AgentOutputError as e:
            raise HTTPException(status_code=502, detail=f"Агент вернул некорректный ответ: {e}") from e
        return DigestResponse.model_validate(asdict(result))

    return app


app = create_app()
