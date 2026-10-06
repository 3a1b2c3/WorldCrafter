#!/usr/bin/env bash
# Run every bundled example under test/I2V/* and test/T2V/* through
# inference.py (27 I2V + 3 T2V = 30 runs). Each one reloads the model
# fresh (inference.py's own design, not something this script works around),
# so expect this to take a while -- run in the background / nohup it.
#
#   ./run_all_examples.sh            # Fast model (default)
#   ./run_all_examples.sh base       # Base model
set -uo pipefail   # no -e: one failed example shouldn't stop the rest

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
else
    WEIGHTS="weights/WorldCrafter-Base"
fi
if [ ! -d "$WEIGHTS" ]; then
    echo "ERROR: weights not found: $WEIGHTS" >&2
    exit 1
fi

OUT_DIR="outputs/all_examples_${MODEL_TYPE}"
mkdir -p "$OUT_DIR"

PASS=0
FAIL=0
FAILED_NAMES=()

run_one() {
    local mode="$1" dir="$2" name
    name="$(basename "$dir")"
    local out="$OUT_DIR/${mode}_${name}.mp4"
    if [ -f "$out" ]; then
        echo "[skip] $mode/$name (already exists)"
        return 0
    fi
    echo "=========================================="
    echo "[$mode] $name"
    echo "=========================================="
    local args=(--model-type "$MODEL_TYPE" --mode "$mode"
                --prompt "$dir/prompt.txt" --camera-path "$dir/camera.npy"
                --output-path "$out")
    if [ "$mode" = "i2v" ]; then
        args+=(--image-path "$dir/image.png")
    fi
    if "$VENV_PY" inference.py "${args[@]}"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_NAMES+=("$mode/$name")
    fi
}

for dir in test/I2V/*/; do
    run_one i2v "${dir%/}"
done
for dir in test/T2V/*/; do
    run_one t2v "${dir%/}"
done

echo
echo "=========================================="
echo "Done. $PASS passed, $FAIL failed."
if [ "$FAIL" -gt 0 ]; then
    echo "Failed: ${FAILED_NAMES[*]}"
fi
echo "Outputs: $OUT_DIR/"
