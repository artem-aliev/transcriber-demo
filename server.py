"""
server.py — Single-process FastAPI WebSocket server with live ASR streaming.

Architecture (D002):
    One FastAPI app serves:
      - ``GET /`` → ``static/index.html`` (browser frontend)
      - ``WS /ws`` → live transcription pipeline via ``Transcriber`` (D004, MEM005)

    WebSocket lifecycle maps 1:1 to ASR streaming lifecycle:
      connect → ``start_streaming()``
      binary frame → ``stream_chunk()``
      disconnect → ``finish_streaming()``

Single-session constraint:
    Only one WebSocket may be active at a time because the Transcriber holds a
    single Qwen3ASRModel.LLM() instance.  If a second WebSocket connects while
    one is active, the server logs a warning but does **not** reject the
    connection — however, the first connection's streaming state will be
    corrupted since both share the same Transcriber.  This is documented in
    the log and is acceptable for a single-user local tool.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from transcriber import Transcriber

# ── Structured logger (same pattern as transcriber.py) ──────────────────────
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
)
logger.addHandler(_handler)


# ── Module-level state ──────────────────────────────────────────────────────
_transcriber: Optional[Transcriber] = None
"""Module-level Transcriber instance.  If model load fails, this stays ``None``
and ``/ws`` returns 503."""

_model_available: bool = False
"""Set to ``True`` after successful model load in the lifespan handler."""

_active_streaming: bool = False
"""Tracks whether a WebSocket streaming session is currently active.
Only one session at a time (single LLM instance)."""


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Transcriber model at startup, clean up on shutdown."""
    global _transcriber, _model_available

    logger.info("Server starting — loading ASR model…")
    try:
        _transcriber = Transcriber()
        _model_available = True
        logger.info("ASR model loaded successfully. Server ready.")
    except (ImportError, RuntimeError) as exc:
        logger.error("Model load failed: %s", exc)
        logger.error(
            "Server will start, but /ws will return 503 until the model "
            "is available."
        )
        _model_available = False
    except Exception:
        logger.error(
            "Unexpected error during model load:\n%s", traceback.format_exc()
        )
        _model_available = False

    yield  # ── Server runs here ──

    logger.info("Server shutting down.")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Transcriber", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    """Serve the browser frontend."""
    return FileResponse("static/index.html")


# Mount static file serving for the browser frontend (T02)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── WebSocket endpoint ──────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_transcribe(ws: WebSocket) -> None:
    """Live transcription WebSocket endpoint.

    Protocol:
        - Client sends binary audio frames (float32, 16 kHz mono PCM).
        - Server returns JSON text frames: ``{"text": "...", "language": "English"}``.
        - On error, server returns: ``{"error": "…"}``.

    Lifecycle:
        connect → ``start_streaming()`` → binary frames → ``stream_chunk()``
        → disconnect → ``finish_streaming()``.
    """
    global _active_streaming, _transcriber, _model_available

    client_host = ws.client.host if ws.client else "unknown"
    client_port = ws.client.port if ws.client else 0

    # ── Check model availability ────────────────────────────────────────
    if not _model_available or _transcriber is None:
        logger.warning(
            "/ws connection from %s:%d rejected — model not available.",
            client_host,
            client_port,
        )
        await ws.accept()
        await ws.send_json({"error": "Model not available"})
        await ws.close(code=1011)
        return

    # ── Accept the WebSocket ─────────────────────────────────────────────
    await ws.accept()
    logger.info(
        "WebSocket connected: %s:%d — streaming session started.",
        client_host,
        client_port,
    )

    # ── Single-session constraint ────────────────────────────────────────
    if _active_streaming:
        logger.warning(
            "Second WebSocket connected while a streaming session is already "
            "active.  Only one session is supported — the first connection's "
            "state may be corrupted."
        )
    _active_streaming = True

    state: Optional[Dict[str, Any]] = None
    chunk_count = 0
    total_text_chars = 0

    try:
        # ── Start streaming ──────────────────────────────────────────────
        state = _transcriber.start_streaming(chunk_size_sec=2.0)
        # vLLM's init_streaming_state returns a dict with a "text" key.
        initial_text = state.get("text", "")
        if initial_text:
            await ws.send_json({"text": initial_text, "language": _transcriber.language})
            logger.info("Initial text sent (%d chars).", len(initial_text))

        # ── Receive loop ─────────────────────────────────────────────────
        while True:
            data = await ws.receive_bytes()

            # Guard: skip zero-length frames
            if len(data) == 0:
                logger.debug(
                    "Skipping zero-length binary frame (chunk %d).",
                    chunk_count + 1,
                )
                continue

            chunk_count += 1

            # Convert bytes to float32 array
            try:
                audio_chunk = np.frombuffer(data, dtype=np.float32)
            except ValueError:
                logger.warning(
                    "Chunk %d: byte payload size %d is not a multiple of "
                    "float32 size — np.frombuffer will handle remainder. "
                    "Skipping frame.",
                    chunk_count,
                    len(data),
                )
                await ws.send_json({"error": "Invalid audio frame: byte size "
                                              "not aligned to float32"})
                continue

            # Guard: skip empty or tiny audio
            if len(audio_chunk) == 0:
                logger.debug(
                    "Skipping empty audio chunk (chunk %d).", chunk_count
                )
                continue

            # Process the chunk through the Transcriber
            try:
                text = _transcriber.stream_chunk(audio_chunk, state)
            except Exception:
                logger.error(
                    "Error processing audio chunk %d:\n%s",
                    chunk_count,
                    traceback.format_exc(),
                )
                await ws.send_json({"error": "Transcription processing error"})
                continue

            # Send partial text back to the browser
            if text:
                total_text_chars = len(text)
                await ws.send_json(
                    {"text": text, "language": _transcriber.language}
                )
                logger.debug(
                    "Chunk %d: sent %d chars.", chunk_count, len(text)
                )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected by client: %s:%d. "
            "Flushing final tokens…",
            client_host,
            client_port,
        )
    except Exception:
        logger.error(
            "Unexpected error during streaming session %s:%d:\n%s",
            client_host,
            client_port,
            traceback.format_exc(),
        )
    finally:
        # ── Finish streaming (flush final tokens) ────────────────────────
        if state is not None and _transcriber is not None:
            try:
                result = _transcriber.finish_streaming(state)
                total_text_chars = len(result.text)
                # Send final text if it differs from last partial
                try:
                    await ws.send_json(
                        {"text": result.text, "language": result.language}
                    )
                except Exception:
                    # Client likely already disconnected — final text is logged.
                    pass
            except Exception:
                logger.error(
                    "Error during finish_streaming:\n%s",
                    traceback.format_exc(),
                )

        _active_streaming = False
        logger.info(
            "Streaming session ended: %s:%d — %d chunks processed, "
            "final text: %d chars.",
            client_host,
            client_port,
            chunk_count,
            total_text_chars,
        )


# ── Health check ────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health endpoint — returns model availability and active status."""
    return {
        "status": "ok",
        "model_available": _model_available,
        "active_streaming": _active_streaming,
    }


# ── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
