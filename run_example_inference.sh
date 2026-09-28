#!/usr/bin/env bash
# Activates the uv-managed venv and runs the bundled Fast i2v example.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/uvenv/.venv/bin/activate"

python inference.py --model-type fast --mode i2v \
  --image-path test/I2V/06_waterfall/image.png \
  --prompt test/I2V/06_waterfall/prompt.txt \
  --camera-path test/I2V/06_waterfall/camera.npy
