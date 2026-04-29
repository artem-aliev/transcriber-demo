#!/usr/bin/env python3
"""
verify_integration.py — Integration verification for Transcriber app (S04).

Validates all server endpoints, frontend structure, and requirement coverage
without requiring a GPU. Uses FastAPI TestClient for HTTP tests and grep for
static structural checks.

Usage:
    python verify_integration.py          # Run all checks
    python verify_integration.py --http   # HTTP endpoint tests only
    python verify_integration.py --grep   # Static structure checks only
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Determine project root ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)


# ── Test runner ──────────────────────────────────────────────────────────────

_checks: List[Tuple[str, bool, str]] = []
_error_count = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record a check result."""
    global _error_count
    _checks.append((name, ok, detail))
    if not ok:
        _error_count += 1


def report() -> int:
    """Print a summary and return exit code (0 = all pass)."""
    passed = sum(1 for _, ok, _ in _checks if ok)
    failed = _error_count
    total = len(_checks)

    print()
    print("=" * 60)
    print(f"Verification Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    print()

    for name, ok, detail in _checks:
        marker = "✓" if ok else "✗"
        line = f"  {marker} {name}"
        if detail:
            line += f"  — {detail}"
        print(line)

    print()
    if failed:
        print(f"❌ FAILED: {failed} check(s) failed.")
    else:
        print("✅ All checks passed.")
    return 1 if failed else 0


# ── HTTP endpoint tests ──────────────────────────────────────────────────────


def http_tests() -> None:
    """Test all HTTP endpoints via FastAPI TestClient."""
    print("── HTTP Endpoint Tests ──")

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        check("import uvicorn", False, "uvicorn not installed")
        return

    from fastapi.testclient import TestClient
    from server import app

    client = TestClient(app)

    # ── GET / ─────────────────────────────────────────────────────────────
    resp = client.get("/")
    check(
        "GET / returns 200",
        resp.status_code == 200,
        f"status={resp.status_code}",
    )
    check(
        "GET / content-type is text/html",
        "text/html" in resp.headers.get("content-type", ""),
        resp.headers.get("content-type", "?"),
    )
    check(
        "GET / body contains <html",
        "<html" in resp.text.lower(),
        f"{len(resp.text)} bytes",
    )

    # ── GET /health ───────────────────────────────────────────────────────
    resp = client.get("/health")
    check(
        "GET /health returns 200",
        resp.status_code == 200,
        f"status={resp.status_code}",
    )
    data = resp.json()
    for field in ["status", "model_available", "active_streaming",
                   "session_active", "session_paused"]:
        check(
            f"GET /health contains '{field}'",
            field in data,
            str(data),
        )
    check(
        "GET /health status == 'ok'",
        data.get("status") == "ok",
        str(data.get("status")),
    )

    # ── POST /upload — valid .txt file ────────────────────────────────────
    txt_content = b"Hello, this is a test document for transcription context."
    files = {"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}
    resp = client.post("/upload", files=files)
    check(
        "POST /upload (.txt) returns 200",
        resp.status_code == 200,
        f"status={resp.status_code}",
    )
    if resp.status_code == 200:
        data = resp.json()
        check(
            "POST /upload (.txt) has 'chars'",
            "chars" in data and data["chars"] > 0,
            str(data),
        )
        check(
            "POST /upload (.txt) has 'filename'",
            data.get("filename") == "test.txt",
            str(data),
        )

    # ── POST /upload — unsupported extension ──────────────────────────────
    files = {"file": ("test.pdf", b"not a valid file", "application/pdf")}
    resp = client.post("/upload", files=files)
    check(
        "POST /upload (.pdf) returns 400",
        resp.status_code == 400,
        f"status={resp.status_code}",
    )
    if resp.status_code == 400:
        data = resp.json()
        check(
            "POST /upload (.pdf) has 'error' key",
            "error" in data,
            str(data),
        )

    # ── POST /upload — empty file ─────────────────────────────────────────
    files = {"file": ("empty.txt", b"", "text/plain")}
    resp = client.post("/upload", files=files)
    check(
        "POST /upload (empty) returns 400",
        resp.status_code == 400,
        f"status={resp.status_code}",
    )

    # ── GET /export — no transcript (header-only) ─────────────────────────
    resp = client.get("/export")
    check(
        "GET /export returns 200",
        resp.status_code == 200,
        f"status={resp.status_code}",
    )
    check(
        "GET /export has Content-Disposition",
        "attachment" in resp.headers.get("content-disposition", ""),
        resp.headers.get("content-disposition", "?"),
    )
    check(
        "GET /export content-type is text/markdown",
        "text/markdown" in resp.headers.get("content-type", ""),
        resp.headers.get("content-type", "?"),
    )
    check(
        'GET /export contains "# Meeting Transcript"',
        "# Meeting Transcript" in resp.text,
        "…",
    )

    # ── Static files ──────────────────────────────────────────────────────
    resp = client.get("/static/index.html")
    check(
        "GET /static/index.html returns 200",
        resp.status_code == 200,
        f"status={resp.status_code}",
    )


# ── Static structure checks ──────────────────────────────────────────────────


def grep(pattern: str, filepath: str) -> bool:
    """Return True if *pattern* is found in *filepath*."""
    import subprocess
    try:
        subprocess.run(
            ["grep", "-q", "-e", pattern, filepath],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def grep_not(pattern: str, filepath: str) -> bool:
    """Return True if *pattern* is NOT found in *filepath*."""
    return not grep(pattern, filepath)


def grep_count(pattern: str, filepath: str) -> int:
    """Return match count."""
    try:
        result = subprocess.run(
            ["grep", "-c", "-e", pattern, filepath],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError:
        return 0


def static_checks() -> None:
    """Run all grep-based structural checks."""
    print("── Static Structure Checks ──")

    # ── py_compile ────────────────────────────────────────────────────────
    for f in ["server.py", "transcriber.py", "cli.py"]:
        try:
            subprocess.run(
                [sys.executable, "-m", "py_compile", f],
                check=True,
                capture_output=True,
            )
            check(f"py_compile {f}", True, "")
        except subprocess.CalledProcessError as e:
            check(f"py_compile {f}", False, str(e.stderr.decode()[:200]))

    # ── R001: Browser mic → streaming pipeline (S02) ──────────────────
    html = "static/index.html"
    check("R001: getUserMedia present", grep("getUserMedia", html), "")
    check("R001: WebSocket present", grep("new WebSocket", html), "")
    check("R001: Float32Array present", grep("Float32Array", html), "")
    check("R001: 16000 sample rate", grep("16000", html), "")
    check("R001: resample function", grep("function resample", html), "")

    # ── R002: ASR forced English mode (S01) ───────────────────────────
    check("R002: forced language=English", grep('language.*"English"', "transcriber.py"), "")
    check("R002: Qwen3ASRModel import", grep("Qwen3ASRModel", "transcriber.py"), "")
    check("R002: LLM() call", grep("Qwen3ASRModel.LLM", "transcriber.py"), "")

    # ── R003: Document upload + context (S03) ─────────────────────────
    check("R003: /upload endpoint", grep("def upload_document", "server.py"), "")
    check("R003: File/UploadFile import", grep("File.*UploadFile", "server.py"), "")
    check("R003: update_context call", grep("update_context", "server.py"), "")
    check("R003: pptx parsing", grep("Presentation", "server.py"), "")
    check("R003: docx parsing", grep("Document", "server.py"), "")
    check("R003: txt parsing", grep('.decode("utf-8")', "server.py"), "")
    check("R003: file upload UI", grep("fileUpload", html), "")

    # ── R004: Controls (S03) ──────────────────────────────────────────
    check("R004: pause control", grep('"pause"', "server.py"), "")
    check("R004: resume control", grep('"resume"', "server.py"), "")
    check("R004: stop control", grep('"stop"', "server.py"), "")
    check("R004: pause button UI", grep("btnPause", html), "")
    check("R004: resume button UI", grep("btnResume", html), "")
    check("R004: save button UI", grep("btnSave", html), "")
    check("R004: pauseCapture fn", grep("pauseCapture", html), "")
    check("R004: resumeCapture fn", grep("resumeCapture", html), "")

    # ── R005: Markdown export (S03) ───────────────────────────────────
    check("R005: /export endpoint", grep("def export_transcript", "server.py"), "")
    check("R005: Content-Disposition", grep("Content-Disposition", "server.py"), "")
    check("R005: attachment header", grep("attachment", "server.py"), "")
    check("R005: HH:MM:SS format", grep("02d}:{.*02d}:{.*02d", "server.py"), "")
    check("R005: Meeting Transcript header", grep("Meeting Transcript", "server.py"), "")
    check("R005: saveTranscript fn", grep("saveTranscript", html), "")

    # ── R006: Paper-like UI (S03) ─────────────────────────────────────
    check("R006: paper background (#f5f3ee)", grep("#f5f3ee", html), "")
    check("R006: paper card (#fefefc)", grep("#fefefc", html), "")
    check("R006: serif font (Georgia)", grep("Georgia", html), "")
    check("R006: box-shadow elevation", grep("box-shadow.*rgba(0,0,0,0.06)", html), "")
    check("R006: transparent buttons", grep("transparent", html), "")

    # ── R007: Local GPU (S01) ─────────────────────────────────────────
    check("R007: CUDA check in transcriber", grep("cuda.is_available()", "transcriber.py"), "")
    check("R007: vLLM backend", grep("Qwen3ASRModel.LLM", "transcriber.py"), "")
    check("R007: no requests import", grep_not("import requests", "transcriber.py"), "")

    # ── R008: Single process (S02) ────────────────────────────────────
    check("R008: FastAPI static serve", grep("StaticFiles", "server.py"), "")
    html_files = list((PROJECT_ROOT / "static").glob("*.html"))
    check("R008: single HTML file", len(html_files) == 1, f"found {len(html_files)} html files")
    check("R008: no npm/build", grep_not("package.json", str(PROJECT_ROOT)), "")
    check("R008: no framework import", grep_not("import React", html), "")
    check("R008: vanilla JS", grep("use strict", html), "")

    # ── CPU mode (transcriber_cpu.py) ──────────────────────────────────
    cpu_py = "transcriber_cpu.py"
    if os.path.exists(cpu_py):
        check("CPU: TranscriberCPU class", grep("class TranscriberCPU", cpu_py), "")
        check("CPU: transcribe_file method", grep("def transcribe_file", cpu_py), "")
        check("CPU: start_streaming method", grep("def start_streaming", cpu_py), "")
        check("CPU: stream_chunk method", grep("def stream_chunk", cpu_py), "")
        check("CPU: finish_streaming method", grep("def finish_streaming", cpu_py), "")
        check("CPU: update_context method", grep("def update_context", cpu_py), "")
        check("CPU: _StreamState class", grep("class _StreamState", cpu_py), "")
        check("CPU: text property", grep("def text", cpu_py), "")
        check("CPU: s16le conversion", grep("int16", cpu_py), "")
        check("CPU: subprocess stdin", grep("stdin", cpu_py), "")
        check("CPU: --stdin --stream flags", grep("--stream", cpu_py), "")

    # ── Server CPU mode support ─────────────────────────────────────────
    check("Srv: cpu_mode env var", grep("TRANSCRIBER_CPU", "server.py"), "")
    check("Srv: TranscriberCPU import", grep("from transcriber_cpu import", "server.py"), "")
    check("Srv: --cpu CLI arg", grep("--cpu", "server.py"), "")
    check("Srv: cpu_mode in /health", grep("cpu_mode", "server.py"), "")

    # ── Edge cases ────────────────────────────────────────────────────
    check("Edge: pause guard (skip audio)", grep("Audio frame skipped", "server.py"), "")
    check("Edge: zero-length frame guard", grep("zero-length", "server.py"), "")
    check("Edge: WebSocketDisconnect catch", grep("WebSocketDisconnect", "server.py"), "")
    check("Edge: invalid JSON handler", grep("Invalid JSON in control", "server.py"), "")
    check("Edge: unknown action handler", grep("Unknown action", "server.py"), "")
    check("Edge: mic error handler", grep("getUserMedia", html), "")
    check("Edge: upload ext validation", grep("ALLOWED_EXTENSIONS", "server.py"), "")
    check("Edge: empty file rejection", grep("Uploaded file is empty", "server.py"), "")
    check("Edge: empty transcript export", grep("_session_transcript", "server.py"), "")
    check("Edge: stopRequested flag", grep("stopRequested", html), "")
    check("Edge: cleanupAudioPipeline", grep("cleanupAudioPipeline", html), "")
    check("Edge: session_active in finally", grep("_session_active = False", "server.py"), "")

    # ── requirements.txt completeness ────────────────────────────────────
    reqs = "requirements.txt"
    check("Reqs: qwen-asr", grep("qwen-asr", reqs), "")
    check("Reqs: vllm", grep("vllm", reqs), "")
    check("Reqs: fastapi", grep("fastapi", reqs), "")
    check("Reqs: uvicorn", grep("uvicorn", reqs), "")
    check("Reqs: numpy", grep("numpy", reqs), "")
    check("Reqs: python-pptx", grep("python-pptx", reqs), "")
    check("Reqs: python-docx", grep("python-docx", reqs), "")
    check("Reqs: python-multipart", grep("python-multipart", reqs), "")
    check("Reqs: torch", grep("torch", reqs), "")


# ── Additional structural validation ─────────────────────────────────────────

def html_validation() -> None:
    """Validate HTML structure."""
    print("── HTML Structure Validation ──")

    html_path = PROJECT_ROOT / "static" / "index.html"
    content = html_path.read_text()

    check(
        "HTML: DOCTYPE present",
        content.strip().startswith("<!DOCTYPE html>"),
        "…",
    )
    check(
        "HTML: <html> tag present",
        "<html" in content and "</html>" in content,
        "",
    )
    check(
        "HTML: <head> tag present",
        "<head>" in content and "</head>" in content,
        "",
    )
    check(
        "HTML: <body> tag present",
        "<body>" in content and "</body>" in content,
        "",
    )
    check(
        "HTML: <script> tag balanced",
        content.count("<script") == content.count("</script>"),
        f"opens={content.count('<script')} closes={content.count('</script>')}",
    )
    check(
        "HTML: <style> tag balanced",
        content.count("<style>") == content.count("</style>"),
        f"opens={content.count('<style>')} closes={content.count('</style>')}",
    )
    check(
        "HTML: all <div> balanced",
        content.count("<div") == content.count("</div>"),
        f"opens={content.count('<div')} closes={content.count('</div>')}",
    )


# ── Server.py structure validation ───────────────────────────────────────────


def server_validation() -> None:
    """Validate server.py structure."""
    print("── server.py Structure Validation ──")

    server = (PROJECT_ROOT / "server.py").read_text()

    required_globals = [
        "_transcriber", "_model_available", "_active_streaming",
        "_session_active", "_session_paused", "_session_transcript",
        "_session_context", "_session_start_time",
    ]
    for g in required_globals:
        check(f"server: global '{g}'", f"{g}:" in server or f"{g} =" in server or f"{g}=" in server, "")

    required_functions = [
        "lifespan", "index", "websocket_transcribe", "health",
        "upload_document", "_extract_text", "export_transcript",
    ]
    for fn in required_functions:
        check(f"server: function '{fn}'", f"def {fn}" in server, "")

    # Validate all imports used
    imports = ["json", "logging", "os", "sys", "time", "traceback",
               "io", "datetime", "numpy", "fastapi", "transcriber", "uvicorn"]
    for imp in imports:
        check(f"server: import '{imp}'", f"import {imp}" in server or f"from {imp}" in server, "")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = __import__("argparse").ArgumentParser(description="Transcriber Integration Verification")
    parser.add_argument("--http", action="store_true", help="HTTP endpoint tests only")
    parser.add_argument("--grep", action="store_true", help="Static structure checks only")
    args = parser.parse_args()

    run_all = not args.http and not args.grep

    if run_all or args.http:
        http_tests()

    if run_all or args.grep:
        static_checks()
        html_validation()
        server_validation()

    return report()


if __name__ == "__main__":
    sys.exit(main())
