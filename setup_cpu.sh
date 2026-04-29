#!/usr/bin/env bash
#
# setup_cpu.sh — Compile antirez/qwen-asr and download Qwen3-ASR-0.6B model.
#
# Usage:
#     bash setup_cpu.sh
#
# This script:
#   1. Clones antirez/qwen-asr (C inference engine)
#   2. Compiles the binary with BLAS support
#   3. Downloads the Qwen3-ASR-0.6B model
#
# After setup, start the server in CPU mode:
#     python server.py --cpu
#

set -euo pipefail

REPO_URL="https://github.com/antirez/qwen-asr.git"
REPO_DIR="qwen-asr"
MODEL_DIR="qwen3-asr-0.6b"
BINARY_PATH="./qwen_asr"

echo "=== Transcriber CPU Setup ==="
echo ""

# ── Step 1: Clone / update qwen-asr ──────────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    echo "[1/3] Updating existing qwen-asr repository…"
    git -C "$REPO_DIR" pull --ff-only 2>/dev/null || echo "  (already up to date or unable to pull)"
else
    echo "[1/3] Cloning antirez/qwen-asr…"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

# ── Step 2: Compile ─────────────────────────────────────────────────────────
echo "[2/3] Compiling qwen_asr binary…"
make -C "$REPO_DIR" blas

# Verify binary exists
if [ ! -f "$REPO_DIR/qwen_asr" ]; then
    echo "ERROR: Compilation failed — qwen_asr binary not found."
    exit 1
fi

echo "  Binary compiled: $REPO_DIR/qwen_asr"

# ── Step 3: Download model ───────────────────────────────────────────────────
if [ -d "$MODEL_DIR" ]; then
    echo "[3/3] Model directory already exists: $MODEL_DIR"
    echo "  (delete it and re-run to re-download)"
else
    echo "[3/3] Downloading Qwen3-ASR-0.6B model…"
    if [ -x "$REPO_DIR/download_model.sh" ]; then
        bash "$REPO_DIR/download_model.sh"
    else
        echo "  Using huggingface-cli to download…"
        pip install -q "huggingface_hub[cli]"
        huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir "$MODEL_DIR"
    fi
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the server in CPU mode:"
echo "  python server.py --cpu --cpu-binary-path $REPO_DIR/qwen_asr"
echo ""
echo "Or test with a WAV file:"
echo "  $REPO_DIR/qwen_asr -d $MODEL_DIR -i samples/jfk.wav --language English"
