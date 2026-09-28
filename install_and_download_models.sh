#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[1/4] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[2/4] Installing ffmpeg..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y ffmpeg
  else
    echo "ffmpeg is required but no apt-get was found on this system. Install FFmpeg manually and rerun this script."
    exit 1
  fi
fi

echo "[3/4] Installing Python dependencies with uv..."
uv sync --project uvenv --extra demo

mkdir -p "$REPO_DIR/weights"

echo "[4/4] Downloading model weights with Hugging Face CLI..."
uv run --project uvenv hf download TencentARC/WorldCrafter-Fast --local-dir "$REPO_DIR/weights/WorldCrafter-Fast"
uv run --project uvenv hf download TencentARC/WorldCrafter-Base --local-dir "$REPO_DIR/weights/WorldCrafter-Base"
uv run --project uvenv hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "$REPO_DIR/weights/Qwen3-VL-4B-Instruct"

echo "Setup complete."
echo "Model files are in:"
echo "  $REPO_DIR/weights/WorldCrafter-Fast"
echo "  $REPO_DIR/weights/WorldCrafter-Base"
echo "  $REPO_DIR/weights/Qwen3-VL-4B-Instruct"
echo ""
echo "To activate the environment:"
echo "  cd $REPO_DIR"
echo "  source uvenv/.venv/bin/activate"
echo ""
echo "To run inference:"
echo "  python inference.py --model-type fast --mode i2v --image-path test/I2V/06_waterfall/image.png --prompt test/I2V/06_waterfall/prompt.txt --camera-path test/I2V/06_waterfall/camera.npy"
