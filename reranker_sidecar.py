from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
import os
import threading
import time
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

LOGGER = logging.getLogger("ombre_reranker_sidecar")
DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8030
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 4
DEFAULT_RETRY_SECONDS = 60

_model = None
_model_name = ""
_model_device = ""
_model_error = ""
_model_loading = False
_model_loaded_at = 0.0
_model_last_attempt_at = 0.0
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def _env_text(name: str, default: str = "") -> str:
    return str(os.environ.get(name) or default).strip()


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _configured_model() -> str:
    return _env_text("OMBRE_LOCAL_RERANKER_MODEL", DEFAULT_MODEL)


def _configured_device() -> str:
    explicit = _env_text("OMBRE_LOCAL_RERANKER_DEVICE")
    if explicit:
        return explicit
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model_sync() -> None:
    global _model, _model_name, _model_device, _model_error, _model_loading, _model_loaded_at, _model_last_attempt_at
    with _model_lock:
        if _model is not None or _model_loading:
            return
        _model_loading = True
        _model_error = ""
        _model_last_attempt_at = time.time()
    model_name = _configured_model()
    device = _configured_device()
    try:
        from sentence_transformers import CrossEncoder

        LOGGER.info("Loading reranker model model=%s device=%s", model_name, device)
        model = CrossEncoder(
            model_name,
            device=device,
            max_length=_env_int("OMBRE_LOCAL_RERANKER_MAX_LENGTH", DEFAULT_MAX_LENGTH, 128, 8192),
        )
        with _model_lock:
            _model = model
            _model_name = model_name
            _model_device = device
            _model_loaded_at = time.time()
        LOGGER.info("Reranker model ready model=%s device=%s", model_name, device)
    except Exception as exc:
        with _model_lock:
            _model_error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Reranker model load failed")
    finally:
        with _model_lock:
            _model_loading = False


def start_model_loading() -> None:
    with _model_lock:
        if _model is not None or _model_loading:
            return
        if _model_error and time.time() - _model_last_attempt_at < DEFAULT_RETRY_SECONDS:
            return
    thread = threading.Thread(target=_load_model_sync, name="ombre-reranker-loader", daemon=True)
    thread.start()


def _health_payload() -> dict[str, Any]:
    with _model_lock:
        loaded = _model is not None
        loading = _model_loading
        error = _model_error
        model_name = _model_name or _configured_model()
        device = _model_device or _env_text("OMBRE_LOCAL_RERANKER_DEVICE") or "auto"
        loaded_at = _model_loaded_at
    return {
        "status": "ok",
        "service": "ombre-reranker",
        "ready": loaded,
        "loading": loading,
        "model": model_name,
        "device": device,
        "error": error or None,
        "loaded_at": loaded_at or None,
    }


def _validate_rerank_body(body: Any) -> tuple[str, list[str], int | None, bool]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    query = str(body.get("query") or "").strip()
    documents = body.get("documents")
    if not query:
        raise ValueError("query is required")
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a non-empty array")
    normalized = [str(item or "") for item in documents]
    if len(normalized) > 100:
        raise ValueError("documents exceeds maximum of 100")
    top_n = body.get("top_n")
    if top_n is not None:
        try:
            top_n = max(1, min(int(top_n), len(normalized)))
        except (TypeError, ValueError) as exc:
            raise ValueError("top_n must be an integer") from exc
    return_documents = bool(body.get("return_documents", False))
    return query, normalized, top_n, return_documents


def _rerank_sync(query: str, documents: list[str], top_n: int | None, return_documents: bool) -> list[dict[str, Any]]:
    with _model_lock:
        model = _model
    if model is None:
        raise RuntimeError("reranker model is not ready")

    import torch

    pairs = [(query, document) for document in documents]
    with _inference_lock:
        scores = model.predict(
            pairs,
            batch_size=_env_int("OMBRE_LOCAL_RERANKER_BATCH_SIZE", DEFAULT_BATCH_SIZE, 1, 32),
            activation_fn=torch.nn.Sigmoid(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    rows = []
    for index, score in enumerate(scores):
        item: dict[str, Any] = {
            "index": index,
            "relevance_score": max(0.0, min(1.0, float(score))),
        }
        if return_documents:
            item["document"] = {"text": documents[index]}
        rows.append(item)
    rows.sort(key=lambda item: item["relevance_score"], reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return rows


async def health(_request: Request) -> JSONResponse:
    start_model_loading()
    return JSONResponse(_health_payload())


async def rerank(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        query, documents, top_n, return_documents = _validate_rerank_body(body)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    start_model_loading()
    with _model_lock:
        ready = _model is not None
        loading = _model_loading
        error = _model_error
    if not ready:
        return JSONResponse(
            {
                "error": "reranker_not_ready",
                "loading": loading,
                "detail": error or None,
            },
            status_code=503,
        )

    try:
        results = await asyncio.to_thread(_rerank_sync, query, documents, top_n, return_documents)
    except Exception as exc:
        LOGGER.exception("Rerank request failed")
        return JSONResponse({"error": "rerank_failed", "detail": str(exc)}, status_code=500)
    return JSONResponse({"model": _configured_model(), "results": results})


@asynccontextmanager
async def lifespan(_app):
    start_model_loading()
    yield


app = Starlette(
    debug=False,
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/rerank", rerank, methods=["POST"]),
        Route("/v1/rerank", rerank, methods=["POST"]),
    ],
    lifespan=lifespan,
)


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, _env_text("OMBRE_LOCAL_RERANKER_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )
    host = _env_text("OMBRE_LOCAL_RERANKER_HOST", DEFAULT_HOST)
    port = _env_int("OMBRE_LOCAL_RERANKER_PORT", DEFAULT_PORT, 1, 65535)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
