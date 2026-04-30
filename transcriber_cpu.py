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
import pty
import select
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from threading import Lock, Thread
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
    master_fd: int  # PTY master fd for reading stdout/stderr
    stderr_lines: List[str]
    _lock: Any  # threading.Lock

    @property
    def text(self) -> str:
        """Compatibility with ASRStreamingState.text."""
        with self._lock:
            return self.accumulated

    def _append_text(self, chunk: str) -> None:
        with self._lock:
            self.accumulated += chunk

    def _replace_text(self, text: str) -> None:
        with self._lock:
            self.accumulated = text

    def kill(self) -> None:
        try:
            self.process.stdin.close()
        except Exception:
            pass
        if hasattr(self, 'master_fd'):
            try:
                os.close(self.master_fd)
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

        Uses a PTY (pseudo-terminal) so the binary uses line-buffered
        stdout instead of fully-buffered pipe mode.  Without a PTY, the
        binary would only flush its output when stdin is closed.

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
            "-S", "2",
            "--language", self._language,
        ]
        if self._context:
            cmd.extend(["--prompt", self._context])

        try:
            process, master_fd, stderr_lines = _start_pty_process(cmd)
        except FileNotFoundError:
            raise RuntimeError(
                f"qwen_asr binary not found at {self._binary_path}. "
                "Build it with: cd qwen-asr && make blas"
            )

        state = _StreamState(
            process=process,
            accumulated="",
            master_fd=master_fd,
            stderr_lines=stderr_lines,
            _lock=Lock(),
        )

        # Background thread: continuously read PTY master into state.accumulated
        def _read_pty() -> None:
            try:
                while True:
                    data = os.read(master_fd, 4096)
                    if not data:
                        break
                    try:
                        text = data.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    if _is_status_line(text):
                        state.stderr_lines.append(text.rstrip())
                    else:
                        state._append_text(text)
            except (OSError, ValueError):
                pass

        Thread(target=_read_pty, daemon=True).start()

        state._total_audio_samples = 0

        return state

    def stream_chunk(
        self,
        audio_chunk: np.ndarray,
        state: _StreamState,
    ) -> str:
        """Process one audio chunk and return the latest partial text.

        Converts float32 audio to s16le, writes to the subprocess stdin.
        Auto-flushes every ~6 seconds of accumulated audio by closing the
        current binary's stdin (forcing output), reading the result, and
        launching a fresh binary with ``--prompt`` for continuity.

        Args:
            audio_chunk: 1-D ``float32`` NumPy array, 16 kHz mono PCM.
            state: The ``_StreamState`` returned by :meth:`start_streaming`.

        Returns:
            The current accumulated transcription text.
        """
        process = state.process
        if process.stdin is None or process.poll() is not None:
            return state.text

        if len(audio_chunk) == 0:
            return state.text

        # Convert float32 → s16le
        clamped = np.clip(audio_chunk, -1.0, 1.0)
        int16_data = (clamped * 32767).astype(np.int16)

        try:
            process.stdin.write(int16_data.tobytes())
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            logger.warning("qwen_asr subprocess stdin closed.")

        state._total_audio_samples += len(audio_chunk)

        # Auto-flush every ~6 seconds of audio (96000 samples @ 16 kHz)
        if state._total_audio_samples >= 96000:
            logger.info("Auto-flushing at %d samples.", state._total_audio_samples)
            self._flush_and_restart(state)

        return state.text

    def _flush_and_restart(self, state: _StreamState) -> None:
        """Close current binary's stdin, read any remaining output,
        and launch a new binary instance with the accumulated text
        as ``--prompt`` for continuity.

        Starts the new binary *before* waiting for the old one, so
        model-loading latency overlaps with the old binary's shutdown."""
        old_process = state.process
        old_master_fd = state.master_fd

        # Close stdin of old binary
        if old_process.stdin is not None:
            try:
                old_process.stdin.close()
            except Exception:
                pass

        # Build new prompt from accumulated text (last ~500 chars)
        flat = state.text.strip()
        prompt = flat[-500:] if len(flat) > 500 else flat
        if prompt:
            self._context = prompt

        # Launch new binary immediately (model loads while old one finishes)
        new_cmd = [
            self._binary_path,
            "-d", self._model_dir,
            "--stdin",
            "--stream",
            "-S", "2",
            "--language", self._language,
        ]
        if self._context:
            new_cmd.extend(["--prompt", self._context])

        try:
            new_process, new_master_fd, new_stderr = _start_pty_process(new_cmd)
        except FileNotFoundError:
            raise RuntimeError(f"qwen_asr binary not found at {self._binary_path}.")

        # Start new reader thread
        def _read_pty() -> None:
            try:
                while True:
                    data = os.read(new_master_fd, 4096)
                    if not data:
                        break
                    try:
                        text = data.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    if _is_status_line(text):
                        state.stderr_lines.append(text.rstrip())
                    else:
                        state._append_text(text)
            except (OSError, ValueError):
                pass

        Thread(target=_read_pty, daemon=True).start()

        # Now read remaining output from OLD binary (non-blocking poll)
        def _drain_old() -> None:
            try:
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    ready, _, _ = select.select([old_master_fd], [], [], 0.3)
                    if not ready:
                        if old_process.poll() is not None:
                            break
                        continue
                    data = os.read(old_master_fd, 4096)
                    if not data:
                        break
                    try:
                        text = data.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    if _is_status_line(text):
                        state.stderr_lines.append(text.rstrip())
                    else:
                        state._append_text(text)
            except (OSError, ValueError):
                pass
            finally:
                try:
                    os.close(old_master_fd)
                except Exception:
                    pass
                try:
                    old_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        old_process.kill()
                    except Exception:
                        pass

        Thread(target=_drain_old, daemon=True).start()

        # Log any stderr from old binary (last lines we collected)
        for line in state.stderr_lines[-5:]:
            if "error" in line.lower() or "fail" in line.lower():
                logger.warning("qwen_asr stderr: %s", line)

        # Update state to point at the new binary
        state.process = new_process
        state.master_fd = new_master_fd
        state.stderr_lines = new_stderr
        state._total_audio_samples = 0

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

        # Read remaining output from PTY master
        try:
            while True:
                ready, _, _ = select.select([state.master_fd], [], [], 1.0)
                if not ready:
                    break
                data = os.read(state.master_fd, 4096)
                if not data:
                    break
                try:
                    text = data.decode("utf-8", errors="replace")
                except Exception:
                    continue
                if _is_status_line(text):
                    state.stderr_lines.append(text.rstrip())
                else:
                    state._append_text(text)
        except (OSError, ValueError):
            pass

        # Close PTY master
        try:
            os.close(state.master_fd)
        except Exception:
            pass

        # Wait for process to exit
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            logger.warning("qwen_asr subprocess did not exit — killing.")
            state.kill()

        elapsed = time.perf_counter() - t_start
        text = state.text.strip()

        logger.info(
            "CPU streaming finalised in %.2f s (%d chars).",
            elapsed,
            len(text),
        )

        # Log any stderr warnings
        for line in state.stderr_lines[-10:]:
            if "error" in line.lower() or "fail" in line.lower():
                logger.warning("qwen_asr stderr: %s", line)

        return TranscriptionResult(text=text, language=self._language)


# ── PTY subprocess helper ────────────────────────────────────────────────────


def _is_status_line(text: str) -> bool:
    """Return True if *text* is a status/diagnostic line, not transcription."""
    lower = text.strip().lower()
    if not lower:
        return True
    if "\x1b[" in text:
        return True
    if lower.startswith("loading"):
        return True
    if lower.startswith("detected:"):
        return True
    if lower.startswith("model loaded"):
        return True
    if lower.startswith("inference:") or lower.startswith("audio:") or lower.startswith("exit:"):
        return True
    if lower.startswith("qwen_mel_spectrogram") or lower.startswith("qwen3-asr"):
        return True
    if lower.startswith("---") or lower.startswith("***"):
        return True
    if "transcription failed" in lower:
        return True
    return False


def _start_pty_process(cmd: list) -> tuple:
    """Launch a subprocess with a PTY for stdout/stderr.

    Returns (process, master_fd, stderr_lines_list).
    """
    master_fd, slave_fd = pty.openpty()

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=slave_fd,
            stderr=slave_fd,
            preexec_fn=os.setsid,
        )
    except Exception:
        os.close(master_fd)
        os.close(slave_fd)
        raise

    os.close(slave_fd)

    return process, master_fd, []


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
