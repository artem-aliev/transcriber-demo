"""
transcriber_cpu.py — CPU-only ASR wrapper using antirez/qwen-asr C binary.

Wraps the compiled ``qwen_asr`` binary as a subprocess, providing the same
API surface as ``Transcriber`` (transcribe_file, start_streaming, stream_chunk,
finish_streaming, update_context) so server.py can swap between GPU (vLLM) and
CPU (qwen_asr binary) backends transparently.

Dependencies (outside the binary):
    - numpy (for float32 ↔ s16le conversion)
    - A compiled ``qwen_asr`` binary (see setup_cpu.sh)
    - Qwen3-ASR-0.6B model files in a local directory

Usage::

    from transcriber_cpu import TranscriberCPU

    t = TranscriberCPU(model_dir="qwen3-asr-0.6b", language="English")
    result = t.transcribe_file("audio.wav")
    print(result.text)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from threading import Thread
from typing import Any, Dict, List, Optional

import numpy as np

# ── Structured logger ────────────────────────────────────────────────────────
logger = logging.getLogger("transcriber_cpu")
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
    text: str
    language: str


@dataclass
class _StreamState:
    """Holds a running qwen_asr subprocess for streaming transcription."""

    process: subprocess.Popen
    accumulated: str
    stderr_lines: List[str]

    @property
    def text(self) -> str:
        """Compatibility with ASRStreamingState.text."""
        return self.accumulated

    def kill(self) -> None:
        try:
            self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.stdout.close()
        except Exception:
            pass
        try:
            self.process.kill()
        except Exception:
            pass
        try:
            self.process.wait(timeout=5)
        except Exception:
            pass


# ── TranscriberCPU ─────────────────────────────────────────────────────────────


class TranscriberCPU:
    """CPU-only ASR engine wrapping the antirez/qwen-asr C binary.

    Loads the model metadata on construction, exposes file-level and
    streaming transcription via subprocess calls to the ``qwen_asr`` binary.

    Args:
        model_dir: Path to the directory containing model safetensors and
            config files (e.g. ``qwen3-asr-0.6b``).
        binary_path: Path to the ``qwen_asr`` binary (default: ``./qwen-asr/qwen_asr``).
        language: Forced output language (default ``"English"``).
        context: Optional domain-hint text fed via ``--prompt``.
    """

    def __init__(
        self,
        model_dir: str = "qwen3-asr-0.6b",
        binary_path: str = "./qwen-asr/qwen_asr",
        language: str = "English",
        context: str = "",
    ) -> None:
        # Resolve paths immediately so they survive chdir (uvicorn, etc.)
        self._binary_path = os.path.abspath(binary_path)
        self._model_dir = os.path.abspath(model_dir)
        self._language = language
        self._context = context

        # Validate binary
        if not os.path.isfile(self._binary_path):
            # Try finding it alongside this file or in PATH
            found = shutil.which(self._binary_path) or shutil.which("qwen_asr")
            if found:
                self._binary_path = found
            else:
                logger.warning(
                    "qwen_asr binary not found at %s. "
                    "Run setup_cpu.sh to compile it.",
                    self._binary_path,
                )

        # Validate model directory
        if not os.path.isdir(self._model_dir):
            logger.warning(
                "Model directory %s not found. "
                "Run setup_cpu.sh to download the model.",
                self._model_dir,
            )

        logger.info(
            "TranscriberCPU ready (binary=%s, model=%s, language=%s).",
            self._binary_path,
            self._model_dir,
            self._language,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return f"Qwen3-ASR-0.6B (CPU via {self._binary_path})"

    @property
    def language(self) -> str:
        return self._language

    @property
    def context(self) -> str:
        return self._context

    @property
    def load_time_seconds(self) -> float:
        return 0.0

    def update_context(self, context: str) -> None:
        self._context = context
        logger.info("Context updated (%d chars).", len(context))

    # ── File transcription ────────────────────────────────────────────────

    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an entire audio file synchronously.

        Args:
            audio_path: Path to a WAV / FLAC / MP3 audio file.

        Returns:
            ``TranscriptionResult`` with the English text and language code.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        t_start = time.perf_counter()
        logger.info("Transcribing file (CPU): %s", audio_path)

        cmd = [
            self._binary_path,
            "-d", self._model_dir,
            "-i", audio_path,
            "--language", self._language,
            "--silent",
        ]
        if self._context:
            cmd.extend(["--prompt", self._context])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            logger.error("Transcription timed out for %s", audio_path)
            return TranscriptionResult(text="", language=self._language)
        except FileNotFoundError:
            raise RuntimeError(
                f"qwen_asr binary not found at {self._binary_path}. "
                "Build it with: cd qwen-asr && make blas"
            )

        if result.returncode != 0:
            logger.error(
                "qwen_asr failed (exit %d): %s",
                result.returncode,
                result.stderr.strip()[-500:],
            )
            return TranscriptionResult(text="", language=self._language)

        text = result.stdout.strip()
        elapsed = time.perf_counter() - t_start
        logger.info(
            "File transcription finished in %.2f s (%d chars).",
            elapsed,
            len(text),
        )

        return TranscriptionResult(text=text, language=self._language)

    # ── Streaming ──────────────────────────────────────────────────────────

    def start_streaming(
        self,
        chunk_size_sec: float = 2.0,
    ) -> _StreamState:
        """Initialise a streaming transcription session.

        Launches the ``qwen_asr`` binary in stdin streaming mode.  Audio
        chunks are written to the subprocess stdin as raw s16le; partial
        text is read from stdout as it becomes available.

        Args:
            chunk_size_sec: Not used by the CPU backend (the binary uses
                its own internal 2-second chunking).

        Returns:
            ``_StreamState`` holding the subprocess handle and accumulated text.
        """
        logger.info(
            "Starting CPU streaming session (language=%s).",
            self._language,
        )

        cmd = [
            self._binary_path,
            "-d", self._model_dir,
            "--stdin",
            "--stream",
            "--language", self._language,
            "--silent",
        ]
        if self._context:
            cmd.extend(["--prompt", self._context])

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # raw bytes for audio
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"qwen_asr binary not found at {self._binary_path}. "
                "Build it with: cd qwen-asr && make blas"
            )

        state = _StreamState(
            process=process,
            accumulated="",
            stderr_lines=[],
        )

        # Start a background thread to capture stderr
        def _read_stderr() -> None:
            if process.stderr is None:
                return
            for line_bytes in process.stderr:
                try:
                    line = line_bytes.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    continue
                state.stderr_lines.append(line)

        Thread(target=_read_stderr, daemon=True).start()

        return state

    def stream_chunk(
        self,
        audio_chunk: np.ndarray,
        state: _StreamState,
    ) -> str:
        """Process one audio chunk and return the latest partial text.

        Converts float32 audio to s16le, writes to the subprocess stdin,
        then reads any available stdout output.

        Args:
            audio_chunk: 1-D ``float32`` NumPy array, 16 kHz mono PCM.
            state: The ``_StreamState`` returned by :meth:`start_streaming`.

        Returns:
            The current partial transcription text.
        """
        process = state.process
        if process.stdin is None or process.poll() is not None:
            return state.accumulated

        # Convert float32 → s16le
        if len(audio_chunk) == 0:
            return state.accumulated

        # Clamp to [-1, 1] and convert to int16
        clamped = np.clip(audio_chunk, -1.0, 1.0)
        int16_data = (clamped * 32767).astype(np.int16)

        try:
            process.stdin.write(int16_data.tobytes())
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            logger.warning("qwen_asr subprocess stdin closed.")
            return state.accumulated

        # Read available stdout (non-blocking via select on Unix)
        if process.stdout is None:
            return state.accumulated

        new_text = ""
        try:
            import select
            while True:
                ready, _, _ = select.select([process.stdout], [], [], 0.0)
                if not ready:
                    break
                # Read a chunk of available data
                data = os.read(process.stdout.fileno(), 4096)
                if not data:
                    break
                new_text += data.decode("utf-8", errors="replace")
        except (OSError, ValueError):
            pass

        if new_text:
            state.accumulated += new_text

        return state.accumulated

    def finish_streaming(self, state: _StreamState) -> TranscriptionResult:
        """Finalise a streaming session and return the complete transcription.

        Closes stdin to signal EOF, waits for the subprocess to finish,
        and reads any remaining output.

        Args:
            state: The ``_StreamState`` from :meth:`start_streaming`.

        Returns:
            ``TranscriptionResult`` with the final, complete text.
        """
        logger.info("Finalising CPU streaming session…")

        process = state.process

        # Close stdin to signal end of audio
        if process.stdin is not None:
            try:
                process.stdin.close()
            except Exception:
                pass

        t_start = time.perf_counter()

        # Read remaining stdout
        if process.stdout is not None:
            try:
                remaining = process.stdout.read()
                if remaining:
                    state.accumulated += remaining.decode("utf-8", errors="replace")
            except Exception:
                pass

        # Wait for process to exit
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            logger.warning("qwen_asr subprocess did not exit — killing.")
            state.kill()

        elapsed = time.perf_counter() - t_start
        text = state.accumulated.strip()

        logger.info(
            "CPU streaming finalised in %.2f s (%d chars).",
            elapsed,
            len(text),
        )

        # Log any stderr warnings
        for line in state.stderr_lines:
            if "error" in line.lower() or "fail" in line.lower():
                logger.warning("qwen_asr stderr: %s", line)

        return TranscriptionResult(text=text, language=self._language)


# ── Module-level convenience ─────────────────────────────────────────────────


def create_transcriber_cpu(
    model_dir: str = "qwen3-asr-0.6b",
    binary_path: str = "./qwen-asr/qwen_asr",
    language: str = "English",
    context: str = "",
) -> TranscriberCPU:
    """Factory function — mirrors ``TranscriberCPU.__init__``."""
    return TranscriberCPU(
        model_dir=model_dir,
        binary_path=binary_path,
        language=language,
        context=context,
    )
