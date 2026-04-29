# Transcriber

Live multilingual meeting transcription and translation — a single-process Python web app that captures microphone audio, transcribes Chinese/Russian/English speech directly to English using Qwen3-ASR-1.7B on local GPUs, displays the transcript live on a white paper-like UI, and exports timestamped markdown.

**No cloud dependencies. All inference runs on local hardware.**

## Prerequisites

- **OS:** Linux (CUDA support)
- **Python:** 3.10–3.12 (PyTorch 2.x does not support 3.13+)
- **GPU:** 2× NVIDIA V100 16GB (Qwen3-ASR-1.7B uses ~3.4 GB VRAM at bf16)
- **CUDA:** Driver and toolkit compatible with PyTorch 2.x
- **Microphone:** Browser-accessible microphone

## Quick Start

```bash
# 1. Clone and set up
git clone <repo>
cd transcriber
bash setup.sh

# 2. Activate venv and start the server
source .venv/bin/activate
python server.py

# 3. Open in browser
open http://localhost:8000
```

## Usage

1. **Start the server:** `python server.py`
2. **Open the app:** Navigate to `http://localhost:8000`
3. **Optional — upload context:** Click "Upload Context" to upload `.pptx`, `.docx`, or `.txt` files with domain-specific vocabulary. This text is passed to the ASR model as a prompt prefix to improve recognition of technical terms.
4. **Start recording:** Click "Start" and grant microphone permission.
5. **Speak:** Talk in Chinese, Russian, or English. The English transcript appears live.
6. **Pause/Resume:** Click "Pause" to temporarily stop recording (transcript is preserved). Click "Resume" to continue.
7. **Stop:** Click "Stop" to end the session.
8. **Save:** Click "Save" to download the transcript as a timestamped markdown file.

## Transcript Format

```markdown
# Meeting Transcript

**Date:** 2026-04-29T12:34:56Z
**Language:** English

---

[00:00:00] Welcome to the meeting.
[00:00:05] Let's discuss the quarterly results.
[00:00:12] We saw strong growth in Q2.
```

Each line has a `[HH:MM:SS]` timestamp relative to the start of recording.

## Architecture

```
Browser (index.html)
  │  getUserMedia() → 16kHz resampling → Float32Array
  │  WebSocket binary frames
  ▼
Server (server.py)  —  FastAPI, single process
  │  POST /upload   →  document text extraction → Transcriber.update_context()
  │  WS /ws         →  start_streaming() → stream_chunk() → finish_streaming()
  │  GET /export    →  timestamped markdown download
  │  GET /health    →  model and session status
  │  GET /          →  static/index.html
  ▼
Transcriber (transcriber.py)
  │  Qwen3ASRModel.LLM() via vLLM backend
  │  Qwen3-ASR-1.7B, forced English output
  │  GPU 0, ~3.4 GB VRAM (bf16)
```

## Verification

Run the integration verification suite:

```bash
source .venv/bin/activate
python verify_integration.py
```

This tests all HTTP endpoints, requirement coverage (R001–R008), edge case handling, HTML structure, server structure, and dependency completeness — 121+ automated checks.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Browser frontend (static HTML) |
| `/ws` | WS | Live transcription pipeline (binary audio → JSON text) |
| `/health` | GET | Server health (model status, session state) |
| `/upload` | POST | Upload document context (.pptx, .docx, .txt) |
| `/export` | GET | Download timestamped markdown transcript |

## File Overview

| File | Purpose |
|---|---|
| `server.py` | FastAPI server with WebSocket, upload, export, health endpoints |
| `transcriber.py` | ASR model wrapper (Qwen3-ASR-1.7B via vLLM) |
| `cli.py` | CLI for file-level transcription |
| `static/index.html` | Browser frontend (mic capture, WebSocket client, paper-like UI) |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Environment setup script |
| `verify_integration.py` | Integration verification suite |
