#!/usr/bin/env bash
# =============================================================================
# IMPORTANT: Python ≤ 3.12 REQUIRED
# PyTorch 2.x does not support Python 3.13+. This script enforces Python 3.10–3.12
# and creates a venv with all dependencies. The target GPU machine must use 3.12.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── 1. Find a compatible Python ──
PYTHON=""
for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | awk '{print $2}')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
            PYTHON="$candidate"
            echo "[setup] Found compatible Python: $PYTHON ($ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: No compatible Python found (needs 3.10, 3.11, or 3.12)." >&2
    echo "Python 3.13+ is NOT supported because PyTorch 2.x has no 3.13 wheels." >&2
    echo "Install Python 3.12 via:" >&2
    echo "  brew install python@3.12        # macOS" >&2
    echo "  sudo apt install python3.12     # Debian/Ubuntu" >&2
    exit 1
fi

# ── 2. Create virtual environment ──
if [ -d "$VENV_DIR" ]; then
    echo "[setup] Removing existing venv at $VENV_DIR ..."
    rm -rf "$VENV_DIR"
fi
echo "[setup] Creating venv with $PYTHON ..."
"$PYTHON" -m venv "$VENV_DIR"

# ── 3. Activate and install dependencies ──
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "[setup] Upgrading pip ..."
pip install --upgrade pip setuptools wheel --quiet
echo "[setup] Installing dependencies from requirements.txt ..."
pip install -r "$SCRIPT_DIR/requirements.txt"
echo "[setup] All dependencies installed."

# ── 4. Verify PyTorch + CUDA ──
echo "[setup] Verifying PyTorch installation ..."
python -c "
import sys, torch
print(f'  Python:       {sys.version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:  {torch.version.cuda}')
    print(f'  GPU count:     {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU[{i}]:         {torch.cuda.get_device_name(i)}')
else:
    print('  WARNING: CUDA not available — transcription will run on CPU (very slow).')
"
echo ""
echo "✓ Setup complete. Activate with: source $VENV_DIR/bin/activate"
