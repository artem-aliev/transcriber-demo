"""
Transcriber — ASR inference wrapper for Qwen3-ASR-1.7B via vLLM backend.

# IMPORTANT: Python ≤ 3.12 REQUIRED (PyTorch 2.x constraint, MEM004)

This module provides the `Transcriber` class — the sole ASR interface consumed by
downstream slices (S02's WebSocket server, S03's web app).  A single
`Qwen3ASRModel.LLM()` instance serves both file-level and streaming transcription
(D004, MEM005), with forced English output (D001).

Architecture decisions encoded here:
  D001 — Qwen3-ASR-1.7B, forced English language mode
  D004 — vLLM backend as single, unified backend
  MEM005 — single model instance for both transcribe_file() and transcribe_stream()
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import numpy as np

# ── Structured logger ────────────────────────────────────────────────────────
logger = logging.getLogger("transcriber")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
)
logger.addHandler(_handler)


# ── Public types ──────────────────────────────────────────────────────────────


@dataclass
class TranscriptionResult:
    """Result of a transcription operation.

    Attributes:
        text: The transcribed / translated text (forced-English unless
            language is overridden).
        language: The output language code (e.g. ``"English"``).
    """

    text: str
    language: str


# ── Transcriber ───────────────────────────────────────────────────────────────


class Transcriber:
    """ASR inference engine wrapping a single Qwen3ASRModel.LLM() instance.

    Loads the model on construction, logs load-time and VRAM details, and
    exposes both file-level and streaming transcription methods.

    Typical usage::

        t = Transcriber()
        result = t.transcribe_file("audio.wav")
        print(result.text)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
        context: str = "",
        language: str = "English",
        gpu_memory_utilization: float = 0.7,
    ) -> None:
        """Load the ASR model via the vLLM backend.

        Args:
            model_name: HuggingFace model ID (D001: Qwen3-ASR-1.7B).
            device: Target device string (validated; vLLM uses
                ``CUDA_VISIBLE_DEVICES`` for actual placement).
            context: Optional domain-hint text fed to the model's internal
                prompt to improve recognition of rare/domain terms (R003).
            language: Forced output language (D001: ``"English"``).
            gpu_memory_utilization: Fraction of GPU memory to reserve
                (0.0–1.0).  The default 0.7 leaves headroom for the OS and
                other processes.

        Raises:
            ImportError: If ``qwen_asr`` is not installed.
            RuntimeError: If CUDA is unavailable or model download fails.
            torch.cuda.OutOfMemoryError: If VRAM allocation fails.
        """
        self._model_name = model_name
        self._device = device
        self._context = context
        self._language = language
        self._gpu_memory_utilization = gpu_memory_utilization

        # ── Validate CUDA availability ───────────────────────────────────
        if device.startswith("cuda"):
            try:
                import torch  # noqa: F811 — used for check only
            except ImportError:
                raise ImportError(
                    "torch is not installed. Install with: "
                    "pip install torch>=2.0.0,<2.6.0"
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. vLLM requires a CUDA-capable GPU. "
                    "On the target machine, verify: "
                    "python -c 'import torch; print(torch.cuda.is_available())'"
                )

        # ── Deferred import — qwen_asr may be heavy ──────────────────────
        try:
            from qwen_asr import Qwen3ASRModel  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "qwen-asr is not installed. Install with: "
                "pip install qwen-asr[vllm]==0.0.6"
            )

        # ── Load model + measure ─────────────────────────────────────────
        t_start = time.perf_counter()
        logger.info(
            "Loading model %s (device=%s, gpu_memory_util=%.2f)…",
            model_name,
            device,
            gpu_memory_utilization,
        )

        self._model: Any = Qwen3ASRModel.LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_inference_batch_size=128,
            max_new_tokens=4096,
        )

        elapsed = time.perf_counter() - t_start
        logger.info("Model loaded in %.2f seconds.", elapsed)

        # ── Log VRAM info ────────────────────────────────────────────────
        if device.startswith("cuda"):
            try:
                import torch  # noqa: F811
            except ImportError:
                pass
            else:
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                logger.info(
                    "VRAM: allocated=%.0f MiB, peak (session)=%.0f MiB",
                    allocated_mb,
                    max_allocated_mb,
                )

        self._load_elapsed = elapsed

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """The HuggingFace model ID in use."""
        return self._model_name

    @property
    def language(self) -> str:
        """Forced output language."""
        return self._language

    @property
    def context(self) -> str:
        """Current domain-context hint."""
        return self._context

    @property
    def load_time_seconds(self) -> float:
        """Model load duration in seconds (from perf_counter)."""
        return self._load_elapsed

    def update_context(self, context: str) -> None:
        """Replace the domain-context hint for subsequent transcriptions.

        The context string is fed to the model's internal prompt and helps
        it recognise rare or domain-specific terms (R003).  It is not shown
        to the end user.

        Args:
            context: Free-form text (document title, keywords, etc.).
        """
        self._context = context
        logger.info("Context updated (%d chars).", len(context))

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an entire audio file synchronously.

        No timestamps are returned — the forced aligner is not used in S01
        (avoids the extra ~1.2 GB VRAM cost).

        Args:
            audio_path: Path to a WAV / FLAC / MP3 audio file.

        Returns:
            ``TranscriptionResult`` with the English text and language code.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
        """
        import os

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t_start = time.perf_counter()
        logger.info("Transcribing file: %s", audio_path)

        results: Any = self._model.transcribe(
            audio=[audio_path],
            language=[self._language],
            context=[self._context] if self._context else None,
            return_time_stamps=False,
        )

        elapsed = time.perf_counter() - t_start
        first: Any = results[0]
        logger.info(
            "File transcription finished in %.2f s (%d chars).",
            elapsed,
            len(first.text),
        )

        return TranscriptionResult(text=first.text, language=first.language)

    def start_streaming(
        self,
        chunk_size_sec: float = 2.0,
    ) -> Dict[str, Any]:
        """Initialise a streaming transcription session.

        Returns a state object that must be passed to ``stream_chunk()``
        and ``finish_streaming()``.  The state is a mutable dictionary
        managed by the vLLM backend.

        Args:
            chunk_size_sec: Duration (seconds) of each audio chunk the
                caller will feed.  The backend uses this together with
                *unfixed_chunk_num* to determine the rollback window.

        Returns:
            A streaming-state dictionary.  Its ``"text"`` key holds the
            latest partial transcription.
        """
        logger.info(
            "Starting streaming session (chunk_size_sec=%.1f, language=%s).",
            chunk_size_sec,
            self._language,
        )

        state: Dict[str, Any] = self._model.init_streaming_state(
            context=self._context if self._context else "",
            language=self._language,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=chunk_size_sec,
        )

        return state

    def stream_chunk(
        self,
        audio_chunk: np.ndarray,
        state: Dict[str, Any],
    ) -> str:
        """Process one audio chunk and return the latest partial text.

        Args:
            audio_chunk: 1-D ``float32`` NumPy array, 16 kHz mono PCM.
            state: The streaming-state dictionary returned by
                :meth:`start_streaming`.

        Returns:
            The current partial transcription text (may be a prefix of the
            final result).
        """
        # vLLM's streaming_transcribe mutates state in-place.
        self._model.streaming_transcribe(audio_chunk, state)
        return state["text"]

    def finish_streaming(self, state: Dict[str, Any]) -> TranscriptionResult:
        """Finalise a streaming session and return the complete transcription.

        **Must be called** after the last chunk — otherwise the final
        token(s) remain uncommitted and the output will be truncated.

        Args:
            state: The streaming-state dictionary from
                :meth:`start_streaming`.

        Returns:
            ``TranscriptionResult`` with the final, complete text.
        """
        logger.info("Finalising streaming session…")

        t_start = time.perf_counter()
        self._model.finish_streaming_transcribe(state)
        elapsed = time.perf_counter() - t_start

        text = state["text"]
        logger.info(
            "Streaming finalised in %.2f s (%d chars).",
            elapsed,
            len(text),
        )

        return TranscriptionResult(text=text, language=self._language)


# ── Module-level convenience ─────────────────────────────────────────────────


def create_transcriber(
    model_name: str = "Qwen/Qwen3-ASR-1.7B",
    device: str = "cuda:0",
    context: str = "",
    language: str = "English",
    gpu_memory_utilization: float = 0.7,
) -> Transcriber:
    """Factory function — mirrors ``Transcriber.__init__`` for discoverability.

    This is a thin wrapper used by ``cli.py`` and integration tests.  Prefer
    direct construction in application code.
    """
    return Transcriber(
        model_name=model_name,
        device=device,
        context=context,
        language=language,
        gpu_memory_utilization=gpu_memory_utilization,
    )
