# Transcriber

Live multilingual meeting transcription — a single-process Python web app that
captures microphone audio, transcribes Chinese / Russian / English speech
directly to English, displays the transcript on a clean paper-like UI, and
exports timestamped markdown.

All inference runs on **local hardware** — no cloud, no API keys, no telemetry.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10–3.12-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

## Features

- **Live transcription** — speech appears in the browser within 2–4 seconds
- **Multilingual** — transcribes Chinese, Russian, English (and 49 more) directly to English
- **Two backends** — GPU (NVIDIA V100, Qwen3-ASR-1.7B via vLLM) or CPU (any x86/ARM, Qwen3-ASR-0.6B via [antirez/qwen-asr](https://github.com/antirez/qwen-asr))
- **Silence-aware chunking** — only sends speech, not background noise
- **Phrase segmentation** — audio is split into natural phrases at silence boundaries
- **Document context** — upload `.pptx`, `.docx`, `.txt` files to prime the model with domain vocabulary
- **Full session control** — Start / Pause / Resume / Stop / Save lifecycle
- **Transcript accumulation** — text builds up across pauses, never lost
- **Markdown export** — timestamped `[HH:MM:SS]` format, one-click download
- **Paper-like UI** — warm white, serif font, outlined buttons — no dark-mode eye strain during long meetings
- **Single process** — no npm, no build step, no separate frontend server

## Quick Start

### CPU mode (any machine)

```bash
git clone https://github.com/artem-aliev/transcriber-demo.git
cd transcriber-demo

# 1. Compile the C inference engine + download Qwen3-ASR-0.6B
bash setup_cpu.sh

# 2. Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install numpy fastapi uvicorn[standard] python-multipart python-docx python-pptx websockets

# 3. Start
python server.py --cpu
```

Open **http://localhost:5000**, click **Start**, speak.

### GPU mode (2× V100 16 GB)

```bash
git clone https://github.com/artem-aliev/transcriber-demo.git
cd transcriber-demo

bash setup.sh                      # creates venv, installs torch + qwen-asr + vllm
source .venv/bin/activate
python server.py                   # loads Qwen3-ASR-1.7B on GPU 0
```

## Usage

| Step | Action |
|---|---|
| 1 | Open `http://localhost:5000` |
| 2 | (Optional) Click **Upload Context** — `.pptx` / `.docx` / `.txt` with project terms |
| 3 | Click **Start** — grant microphone permission |
| 4 | Speak in any supported language — English transcript appears live |
| 5 | Click **Pause** to silence recording (transcript is preserved) |
| 6 | Click **Resume** to continue — new text appends below |
| 7 | Click **Stop** to finalize — remaining audio is flushed and transcribed |
| 8 | Click **Save** — downloads `transcript-YYYY-MM-DD.md` |

### Server CLI

```
python server.py [--cpu] [--cpu-model-dir DIR] [--cpu-binary-path PATH]
                 [--host HOST] [--port PORT]
```

| Flag | Default | Description |
|---|---|---|
| `--cpu` | off | Use CPU backend (antirez/qwen-asr) |
| `--cpu-model-dir` | `qwen3-asr-0.6b` | Path to model directory |
| `--cpu-binary-path` | `./qwen-asr/qwen_asr` | Path to compiled binary |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `5000` | Bind port |

Environment variables (`TRANSCRIBER_CPU`, `TRANSCRIBER_CPU_MODEL_DIR`,
`TRANSCRIBER_CPU_BINARY`) can be used instead of flags.

## WebSocket Protocol

The browser streams audio and control messages over a single WebSocket
connection at `/ws`.

### Browser → Server

| Frame type | Content | Purpose |
|---|---|---|
| Binary | `Float32Array` (16 kHz, mono PCM) | Audio chunk (only speech — silence is filtered client-side) |
| Text (JSON) | `{"action":"pause"}` | Pause transcription |
| Text (JSON) | `{"action":"resume"}` | Resume transcription |
| Text (JSON) | `{"action":"stop"}` | Stop session, flush remaining audio |

### Server → Browser

| Message | Example | When |
|---|---|---|
| Partial text | `{"text":"Hello world","language":"English"}` | Every ~0.5 s while ASR produces output |
| Paused | `{"status":"paused","text":"..."}` | After `pause` action processed |
| Resumed | `{"status":"resumed"}` | After `resume` action processed |
| Stopped | `{"status":"stopped","chunks":12,"segments":2}` | After `stop` action processed |
| Error | `{"error":"Model not available"}` | Model unavailable, invalid JSON, etc. |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Browser (static/index.html)                             │
│                                                         │
│  getUserMedia() → AudioContext 44.1 kHz                 │
│  └─ ScriptProcessorNode                                 │
│     └─ resample → 16 kHz mono                           │
│        └─ RMS silence detection → phrase segmentation   │
│           └─ WebSocket binary frames (Float32Array)     │
│                                                         │
│  WebSocket text frames (pause / resume / stop)          │
│  Transcript display (accumulated, auto-scroll)          │
│  File upload (pptx / docx / txt → POST /upload)         │
│  Save button (GET /export → .md download)               │
└──────────────────────┬──────────────────────────────────┘
                       │ ws://host:5000/ws
┌──────────────────────▼──────────────────────────────────┐
│ Server (server.py) — FastAPI, single process            │
│                                                         │
│  Lifespan: load Transcriber or TranscriberCPU           │
│  GET  /          → static/index.html                    │
│  WS   /ws        → start_streaming → stream_chunk →    │
│                    finish_streaming                     │
│                    + pause / resume / stop dispatch     │
│  POST /upload    → text extraction → update_context     │
│  GET  /export    → timestamped markdown                 │
│  GET  /health    → model + session status               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ ASR Backend                                            │
│                                                         │
│  GPU: transcriber.py                                    │
│    Qwen3ASRModel.LLM() via vLLM                         │
│    Qwen3-ASR-1.7B, bf16, ~3.4 GB VRAM                  │
│                                                         │
│  CPU: transcriber_cpu.py                                │
│    Subprocess: qwen_asr --stdin --stream                │
│    Background thread reads stdout tokens                │
│    float32 → s16le conversion                           │
│    Qwen3-ASR-0.6B, ~3 GB RAM                            │
└─────────────────────────────────────────────────────────┘
```

## Supported Languages

Qwen3-ASR transcribes speech from **52 languages and dialects** to English
(forced output mode) or to the detected language. Forced English is the default
in this app, matching the multilingual-to-English meeting use case.

Chinese (zh), English (en), Cantonese (yue), Arabic (ar), German (de),
French (fr), Spanish (es), Portuguese (pt), Indonesian (id), Italian (it),
Korean (ko), Russian (ru), Thai (th), Vietnamese (vi), Japanese (ja),
Turkish (tr), Hindi (hi), Malay (ms), Dutch (nl), Swedish (sv), Danish (da),
Finnish (fi), Polish (pl), Czech (cs), Filipino (fil), Persian (fa),
Greek (el), Romanian (ro), Hungarian (hu), Macedonian (mk) + 22 Chinese dialects.

## Export Format

```markdown
# Meeting Transcript

**Date:** 2026-04-29T14:30:00Z
**Language:** English
**Context:** project Q3 roadmap, PostgreSQL migration

---

[00:00:00] Good morning everyone. Let's start the Q3 planning session.
[00:00:08] First item on the agenda is the database migration.
[00:00:15] We need to move from MySQL to PostgreSQL by August.
```

## API

| Method | Path | Input | Output |
|---|---|---|---|
| `GET` | `/` | — | `text/html` (frontend) |
| `WS` | `/ws` | binary audio / JSON controls | JSON text + status |
| `GET` | `/health` | — | `{"status":"ok","model_available":true,"session_active":false,"cpu_mode":false}` |
| `POST` | `/upload` | `multipart/form-data` (`.pptx`, `.docx`, `.txt`) | `{"status":"ok","chars":1234,"filename":"slides.pptx"}` |
| `GET` | `/export` | — | `text/markdown` (attachment download) |

## Verification

```bash
source .venv/bin/activate
python verify_integration.py
```

Runs 130+ automated checks: HTTP endpoints, requirement coverage (R001–R008),
edge cases, HTML structure, server structure, dependency completeness.

## Project Files

```
transcriber-demo/
├── server.py              FastAPI server (WebSocket, upload, export, health)
├── transcriber.py         GPU backend (Qwen3-ASR-1.7B via vLLM)
├── transcriber_cpu.py     CPU backend (antirez/qwen-asr subprocess wrapper)
├── cli.py                 CLI for file transcription
├── static/
│   └── index.html         Browser frontend (mic capture, silence detection,
│                           phrase segmentation, paper-like UI)
├── requirements.txt       Python dependencies (GPU mode)
├── setup.sh               GPU environment setup
├── setup_cpu.sh           CPU environment setup (compiles C binary, downloads model)
├── verify_integration.py  Integration verification suite (130+ checks)
├── LICENSE                Apache 2.0
└── README.md
```

## Prerequisites Detail

### GPU mode

| Requirement | Details |
|---|---|
| OS | Linux with CUDA drivers |
| Python | 3.10–3.12 (PyTorch 2.x constraint) |
| GPU | 2× NVIDIA V100 16 GB or equivalent |
| VRAM | ~3.4 GB at bf16 (single model instance) |
| CUDA | Toolkit compatible with PyTorch ≥2.0 |

### CPU mode

| Requirement | Details |
|---|---|
| OS | macOS (Apple Silicon or Intel) or Linux (x86_64) |
| Python | 3.10+ |
| RAM | ~3 GB for 0.6B model weights + runtime buffers |
| Compiler | `gcc` or `clang` + `make` |
| BLAS | Accelerate framework (macOS, built-in) or OpenBLAS (`apt install libopenblas-dev` on Linux) |

### Both modes

| Requirement | Details |
|---|---|
| Browser | Chrome, Firefox, Edge, Safari (Web Audio API + WebSocket) |
| Microphone | Built-in or external, browser-accessible |

## How It Works

### Frontend audio pipeline

1. `getUserMedia` captures microphone at native sample rate (typically 44.1 kHz)
2. `ScriptProcessorNode` fires every ~10 ms with a 4096-sample buffer
3. Linear interpolation resamples to 16 kHz mono float32
4. **RMS energy** computed per buffer — values below 0.006 are classified as silence
5. Speech accumulates in a phrase buffer; when **750 ms of continuous silence** is
   detected, the accumulated phrase is sent as one binary WebSocket frame
6. Phrases are capped at 20 seconds (forced flush to prevent unbounded buffering)
7. On Pause / Stop: no new audio is captured or sent

### Server-side streaming (GPU)

- `start_streaming()` initializes a vLLM streaming state
- Each binary frame → `np.frombuffer` → `stream_chunk()` → `state.text` updated
- Partial text sent to browser every time it changes
- `finish_streaming()` on disconnect / pause / stop flushes final tokens

### Server-side streaming (CPU)

- `start_streaming()` spawns `qwen_asr --stdin --stream --silent` subprocess
- Audio written to subprocess stdin as raw s16le int16
- Background thread continuously reads stdout tokens into `state.accumulated`
- Text updates are sent to the browser as soon as the background thread picks
  them up (no polling delay)
- `finish_streaming()` closes stdin, waits for subprocess exit, reads remaining
  output

## Known Limitations

- **GPU mode** requires CUDA — no MPS / ROCm support.
- **CPU mode** uses the 0.6B model which is less accurate than the 1.7B GPU model.
- **Single session** — only one WebSocket at a time (one LLM instance). Concurrent
  connections log a warning but are not rejected.
- **Streaming latency** — the `qwen_asr` C binary processes audio in 2-second
  internal chunks; partial output may lag by 2–4 seconds behind speech.
- **Timestamp accuracy** — exported markdown timestamps are wall-clock relative,
  not word-level alignment. For precise timestamps, a forced aligner would be
  needed (not included).
