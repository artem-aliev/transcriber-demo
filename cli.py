#!/usr/bin/env python3
"""CLI entrypoint for smoke-testing Qwen3-ASR-1.7B transcription.

This is the executable proof of S01 — it loads the model, transcribes a
WAV file, and reports the result with timing.  Designed for the target
GPU machine (Python ≤ 3.12, CUDA).

Usage::

    python cli.py --audio speech.wav
    python cli.py --audio speech.wav --language English --context "medical terms"

The ``--help`` flag and ``py_compile`` are the local verification surface
on the dev machine (which cannot install qwen-asr on Python 3.14).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from transcriber import Transcriber, TranscriptionResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description=(
            "Smoke-test Qwen3-ASR-1.7B: transcribe an audio file to English "
            "text and report timing.  Requires a CUDA-capable GPU and Python "
            "≤ 3.12."
        ),
    )

    parser.add_argument(
        "--audio",
        required=True,
        metavar="PATH",
        help="Path to WAV (or FLAC/MP3) audio file to transcribe.",
    )

    parser.add_argument(
        "--language",
        default="English",
        metavar="LANG",
        help="Forced output language (default: English).",
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        metavar="NAME",
        help="HuggingFace model ID (default: Qwen/Qwen3-ASR-1.7B).",
    )

    parser.add_argument(
        "--context",
        default="",
        metavar="TEXT",
        help="Optional domain-context hint to improve recognition of rare terms.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Parse args, load model, transcribe, print result."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── 1. Load model ────────────────────────────────────────────────────
    try:
        transcriber = Transcriber(
            model_name=args.model,
            device="cuda:0",
            context=args.context,
            language=args.language,
        )
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    # OOM reports torch.cuda.OutOfMemoryError when torch is available;
    # otherwise RuntimeError from the model loader.
    except Exception as exc:
        # Catch-all for torch.cuda.OutOfMemoryError and model-download
        # failures from transformers / huggingface_hub.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    load_time = transcriber.load_time_seconds

    # ── 2. Transcribe ────────────────────────────────────────────────────
    t_start = time.perf_counter()

    try:
        result: TranscriptionResult = transcriber.transcribe_file(args.audio)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: Transcription failed — {exc}", file=sys.stderr)
        sys.exit(1)

    transcribe_time = time.perf_counter() - t_start

    # ── 3. Report ────────────────────────────────────────────────────────
    print(f"Language:      {result.language}")
    print(f"Transcription: {result.text}")
    print(f"Load time:     {load_time:.2f} s")
    print(f"Transcribe:    {transcribe_time:.2f} s")


if __name__ == "__main__":
    main()
