#!/usr/bin/env python3
"""Persistent WorldCrafter inference server, speaking the line-delimited-JSON
protocol over stdin/stdout that WBench's worldcrafter_model.py already
expects (see that file's _ensure_server()/generate_with_poses()) -- this
script was referenced there but did not exist in this checkout.

Loads the WorldCrafter diffusion model once (same from_pretrained() call as
inference.py's one-shot CLI), then serves one video per JSON request line
on stdin, replying with one JSON line on stdout per request, until stdin
closes. This is what actually avoids reloading the model per WBench case --
the whole point of the integration, per worldcrafter_model.py's own
docstring.

Protocol
========
Startup: once the model is loaded, prints exactly one line:
    {"ready": true}

Per request, reads one line from stdin:
    {
      "mode": "i2v",
      "image_path": "...", "camera_path": "...", "output_path": "...",
      "prompt": "..." | "auto-first-person" | "auto-third-person",
      "negative_prompt": "..." | null,
      "camera_x_fov": 100.0, "camera_xi": 0.0,
      "seed": 0, "num_inference_steps": null
    }
and writes exactly one response line:
    {"ok": true}
    {"ok": false, "error": "..."}

A per-request failure is caught and reported as {"ok": false, ...} without
exiting -- the caller (generate_with_poses) only treats a closed pipe as
fatal, so later cases must still be servable after an earlier one fails.

Usage:
    python serve_inference.py --model-type fast --model-path weights/WorldCrafter-Fast --seed 0

UNTESTED end-to-end (no WorldCrafter/uvenv venv or weights available in the
environment this was written in) -- verify against a real request before
relying on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from worldcrafter.caption import AUTO_PROMPTS, DEFAULT_CAPTION_MODEL, generate_caption  # noqa: E402

DEFAULT_NEGATIVE_PROMPT_PATH = ROOT / "test" / "negative_prompt.txt"


def build_parser() -> argparse.ArgumentParser:
    # Mirrors worldcrafter/cli.py's model-loading-relevant flags (the
    # request-level flags there -- --camera-path, --prompt, etc. -- are
    # per-request here instead, not startup args).
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("base", "fast"), default="base")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-fov-h-deg", type=float, default=100.0)
    parser.add_argument("--memory-fov-v-deg", type=float, default=71.13349068444832)
    parser.add_argument("--memory-fov-samples-per-axis", type=int, default=10)
    parser.add_argument(
        "--attention-backend",
        choices=("native", "auto", "flash_hub", "_flash_3_hub"),
        default="native",
    )
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--caption-model", default=None)
    parser.add_argument("--negative-prompt-path", type=Path,
                        default=DEFAULT_NEGATIVE_PROMPT_PATH)
    return parser


def _resolve_caption_model(caption_model: str | None) -> str:
    if caption_model:
        return caption_model
    local = ROOT / "weights" / "Qwen3-VL-4B-Instruct"
    return str(local) if local.is_dir() else DEFAULT_CAPTION_MODEL


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"file is empty: {path}")
    return value


def _resolve_prompt(request: dict, args: argparse.Namespace) -> str:
    """Mirrors worldcrafter/caption.py's prepare_prompt(), minus the
    resume-from-disk / state-dir bookkeeping that only matters for the
    one-shot CLI's own output layout -- the server always generates a
    fresh caption for auto-* prompts, same as the original comment in
    worldcrafter_model.py's _ensure_server() already documents as by-design
    (generate_caption() loads and unloads Qwen3-VL per call on its own).
    """
    prompt = request.get("prompt") or ""
    style = AUTO_PROMPTS.get(prompt)
    if style is None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("request 'prompt' must not be empty")
        return prompt

    from diffusers.utils import load_image

    image_path = request.get("image_path")
    if not image_path:
        raise ValueError(f"prompt={prompt!r} requires 'image_path'")
    image = load_image(str(image_path)).resize((args.width, args.height))
    print(f"[serve_inference] generating {prompt} prompt with "
         f"{_resolve_caption_model(args.caption_model)}", file=sys.stderr)
    result = generate_caption(
        image, style, _resolve_caption_model(args.caption_model), args.device)
    return result["prompt"]


def _resolve_negative_prompt(request: dict, args: argparse.Namespace) -> str:
    negative_prompt = request.get("negative_prompt")
    if negative_prompt:
        return negative_prompt
    # worldcrafter_model.py's generate_with_poses() always sends
    # negative_prompt=None (camera-conditioned models have no real negative
    # prompt to offer) -- fall back to the same default the one-shot CLI
    # uses (cli.py's parse_args(): args.negative_prompt = _read_text(
    # args.negative_prompt_path)) rather than passing None through to
    # generate(), which requires a str.
    return _read_text(args.negative_prompt_path)


def main() -> None:
    args = build_parser().parse_args()

    from worldcrafter.inference import WorldCrafter

    model = WorldCrafter.from_pretrained(
        args.model_path,
        model_type=args.model_type,
        device=args.device,
        height=args.height,
        width=args.width,
        seed=args.seed,
        memory_fov_h_deg=args.memory_fov_h_deg,
        memory_fov_v_deg=args.memory_fov_v_deg,
        memory_fov_samples_per_axis=args.memory_fov_samples_per_axis,
        attention_backend=args.attention_backend,
        enable_compile=args.enable_compile,
    )

    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"bad JSON request: {exc}"}),
                 flush=True)
            continue

        try:
            prompt = _resolve_prompt(request, args)
            negative_prompt = _resolve_negative_prompt(request, args)
            mode = request.get("mode", "i2v")
            image_path = request.get("image_path")
            model.generate(
                mode=mode,
                image_path=Path(image_path) if image_path else None,
                camera_path=Path(request["camera_path"]),
                output_path=Path(request["output_path"]),
                prompt=prompt,
                negative_prompt=negative_prompt,
                camera_x_fov=request.get("camera_x_fov", 100.0),
                camera_xi=request.get("camera_xi", 0.0),
                seed=request.get("seed", args.seed),
                num_inference_steps=request.get("num_inference_steps"),
            )
            print(json.dumps({"ok": True}), flush=True)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the server
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}),
                 flush=True)


if __name__ == "__main__":
    main()
