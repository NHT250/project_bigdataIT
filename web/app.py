"""The TechPulse website: three pages and the JSON they read."""
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints

from rag import answering, ask as rag_ask, settings
from web import queries

STATIC = Path(__file__).parent / "static"


class Question(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


def create_app(connect, ask, warm_up=None):
    @asynccontextmanager
    async def lifespan(_app):
        if warm_up:
            threading.Thread(target=warm_up, daemon=True).start()
        yield

    app = FastAPI(title="TechPulse", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def page(name):
        def serve():
            return FileResponse(STATIC / name)
        return serve

    app.get("/", include_in_schema=False)(page("index.html"))
    app.get("/chat", include_in_schema=False)(page("chat.html"))
    app.get("/search", include_in_schema=False)(page("search.html"))

    @app.get("/api/feed")
    def feed(
        source: Literal["all", "arxiv", "hn"] = "all",
        limit: Annotated[int, Query(ge=1, le=100)] = 40,
    ):
        with connect() as conn:
            return {"articles": queries.feed(conn, source, limit)}

    @app.get("/api/search")
    def search(
        q: Annotated[str, Query(max_length=200)] = "",
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        with connect() as conn:
            return {"results": queries.search(conn, q, limit)}

    @app.get("/api/stats")
    def stats():
        with connect() as conn:
            return queries.stats(conn)

    @app.post("/api/chat")
    def chat(body: Question):
        try:
            return asdict(ask(body.question))
        except answering.MissingApiKey as error:
            raise HTTPException(503, str(error)) from error
        except answering.AnswerError as error:
            raise HTTPException(502, str(error)) from error

    return app


@contextmanager
def _connect():
    with psycopg.connect(settings.database_url(), autocommit=True) as conn:
        yield conn


_embedder_lock = threading.Lock()
_embedder = None


def _shared_embedder():
    """Load once: the model takes seconds to load and every question needs it."""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from rag.embedder import Embedder

            _embedder = Embedder()
        return _embedder


def _ask(question):
    api_key = settings.get("GEMINI_API_KEY")
    answering.require_api_key(api_key)
    return rag_ask.run(question, api_key, embedder=_shared_embedder())


app = create_app(connect=_connect, ask=_ask, warm_up=_shared_embedder)
