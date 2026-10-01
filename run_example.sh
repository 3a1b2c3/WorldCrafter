#!/usr/bin/env bash
# Run WorldCrafter's own bundled i2v example, to sanity-check the env/weights
# setup before touching WBench at all. Linux only (per README) -- run inside
# WSL or on a Linux box, not native Windows.
#
#   ./run_example.sh            # Fast model (default)
#   ./run_example.sh base       # Base model
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL_TYPE="${1:-fast}"
VENV_PY="$HERE/uvenv/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: $VENV_PY not found." >&2
    echo "Run first: uv sync --project uvenv --frozen --extra demo" >&2
    exit 1
fi

if [ "$MODEL_TYPE" = "fast" ]; then
    WEIGHTS="weights/WorldCrafter-Fast"
    EXAMPLE_DIR="test/I2V/03_waterfall"
else
    WEIGHTS="weights/WorldCrafter-Base"
    EXAMPLE_DIR="test/I2V/00_cat_robot_vacuum"
fi

if [ ! -d "$WEIGHTS" ]; then
    echo "ERROR: weights not found: $WEIGHTS" >&2
    echo "Run first: hf download TencentARC/WorldCrafter-$([ "$MODEL_TYPE" = fast ] && echo Fast || echo Base) --local-dir $WEIGHTS" >&2
    exit 1
fi

mkdir -p outputs
OUT="outputs/example_${MODEL_TYPE}.mp4"

echo "=========================================="
echo "WorldCrafter example ($MODEL_TYPE)"
echo "=========================================="
"$VENV_PY" inference.py \
  --model-type "$MODEL_TYPE" \
  --mode i2v \
  --image-path "$EXAMPLE_DIR/image.png" \
  --prompt "$EXAMPLE_DIR/prompt.txt" \
  --camera-path "$EXAMPLE_DIR/camera.npy" \
  --output-path "$OUT"

echo
echo "Done. Output: $OUT"
