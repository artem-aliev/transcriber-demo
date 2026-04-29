# Transcriber

Live multilingual meeting transcription and translation — a single-process Python web app that captures microphone audio, transcribes Chinese/Russian/English speech directly to English, displays the transcript live on a white paper-like UI, and exports timestamped markdown.

**No cloud dependencies. All inference runs on local hardware.**

Two backends available:
- **GPU**: Qwen3-ASR-1.7B via vLLM (2× V100 16GB)
- **CPU**: Qwen3-ASR-0.6B via [antirez/qwen-asr](https://github.com/antirez/qwen-asr) (pure C, BLAS-accelerated)

## Prerequisites

### GPU Mode
- **OS:** Linux (CUDA support)
- **Python:** 3.10–3.12 (PyTorch 2.x does not support 3.13+)
- **GPU:** 2× NVIDIA V100 16GB
- **CUDA:** Driver and toolkit compatible with PyTorch 2.x

### CPU Mode
- **OS:** macOS or Linux
- **Python:** 3.10+ (numpy only)
- **RAM:** ~3 GB for Qwen3-ASR-0.6B model
- **Compiler:** gcc/clang + make (for building qwen_asr binary)
- **BLAS:** Accelerate (macOS, built-in) or OpenBLAS (Linux, `apt install libopenblas-dev`)

### Both Modes
- **Microphone:** Browser-accessible microphone

## Quick Start

### GPU Mode

```bash
git clone <repo>
cd transcriber
bash setup.sh
source .venv/bin/activate
python server.py
# Open http://localhost:5000
```

### CPU Mode

```bash
git clone <repo>
cd transcriber

# 1. Compile the C inference engine and download model
bash setup_cpu.sh

# 2. Install numpy (CPU mode has no other Python deps)
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
pip install numpy fastapi uvicorn python-multipart python-docx python-pptx

# 3. Start in CPU mode
python server.py --cpu
# Open http://localhost:5000
```

## Usage

1. **Start the server:** `python server.py` (or `python server.py --cpu`)
2. **Open the app:** Navigate to `http://localhost:5000`
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
| `/health` | GET | Server health (model status, session state, cpu_mode) |
| `/upload` | POST | Upload document context (.pptx, .docx, .txt) |
| `/export` | GET | Download timestamped markdown transcript |

## File Overview

| File | Purpose |
|---|---|
| `server.py` | FastAPI server with WebSocket, upload, export, health endpoints |
| `transcriber.py` | ASR model wrapper (Qwen3-ASR-1.7B via vLLM, GPU) |
| `transcriber_cpu.py` | ASR model wrapper (Qwen3-ASR-0.6B via qwen_asr binary, CPU) |
| `cli.py` | CLI for file-level transcription |
| `static/index.html` | Browser frontend (mic capture, WebSocket client, paper-like UI) |
| `requirements.txt` | Python dependencies (GPU mode) |
| `setup.sh` | GPU environment setup script |
| `setup_cpu.sh` | CPU environment setup script (compiles qwen_asr, downloads model) |
| `verify_integration.py` | Integration verification suite |
