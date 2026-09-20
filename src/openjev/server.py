"""FastAPI server exposing the TypeSafe-compatible ``/v1/systemone`` endpoint.

Point the official SDK at it with ``TYPESAFE_BASE_URL=http://localhost:8000``.
Any bearer token is accepted unless ``OPENJEV_API_KEY`` is set.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .engine import SystemOneEngine
from .schema import ModelMetadata, ModelMetadataList, SystemOneRequest, SystemOneResponse

MODEL_ALIASES = ("jev-latest", "jev-preview")


def _auth(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("OPENJEV_API_KEY")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid API key")


def create_app(engine: SystemOneEngine, *, aliases: tuple[str, ...] = MODEL_ALIASES) -> FastAPI:
    app = FastAPI(title="OpenJev System One API", version="0.1.0")
    app.state.engine = engine
    known = {engine.model_name, *aliases}

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "model": engine.model_name, "backend": getattr(engine.backend, "name", "?")}

    @app.get("/v1/models", response_model=ModelMetadataList, dependencies=[Depends(_auth)])
    def list_models() -> ModelMetadataList:
        today = _dt.date.today().isoformat()
        models = [ModelMetadata(name=engine.model_name, description="OpenJev System One model.", release_date=today)]
        models += [
            ModelMetadata(name=alias, description=f"Alias resolving to {engine.model_name}.", release_date=today)
            for alias in aliases
        ]
        return ModelMetadataList(models=models)

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(_auth)])
    def system_one(request: SystemOneRequest) -> SystemOneResponse:
        if request.model not in known:
            raise HTTPException(
                status_code=404,
                detail=f"unknown model {request.model!r}; use one of {sorted(known)}",
            )
        return engine.evaluate(request)

    @app.exception_handler(RuntimeError)
    async def _runtime_error(_: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app
