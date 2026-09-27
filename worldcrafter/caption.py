from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path


DEFAULT_CAPTION_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
AUTO_PROMPTS = {
    "auto-first-person": "first_person",
    "auto-third-person": "third_person",
}


def generate_caption(image, style: str, model_path: str, device: str) -> dict:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    template = (Path(__file__).parent / "prompts" / f"{style}.txt").read_text(encoding="utf-8")
    target = torch.device(device)
    model = processor = inputs = generated = None
    devices = []
    if target.type == "cuda":
        devices = [target.index if target.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            processor = AutoProcessor.from_pretrained(model_path)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if target.type == "cuda" else torch.float32,
                attn_implementation="sdpa",
            ).to(target)
            model.eval()
            messages = [
                {"role": "system", "content": [{"type": "text", "text": template}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Write the video prompt for this image."},
                    ],
                },
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(target)
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            prompt = processor.decode(
                generated[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            if generated[0, -1].item() != processor.tokenizer.eos_token_id:
                raise RuntimeError("Caption reached the token limit before finishing")
            if not prompt:
                raise RuntimeError("Caption model returned an empty prompt")
            return {
                "prompt": " ".join(prompt.split()),
                "model": model_path,
                "model_revision": getattr(model.config, "_commit_hash", None),
                "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "style": style,
            }
    finally:
        del generated, inputs, model, processor
        gc.collect()
        if target.type == "cuda":
            with torch.cuda.device(target):
                torch.cuda.empty_cache()


def prepare_prompt(args) -> None:
    style = AUTO_PROMPTS.get(args.prompt)
    if style is None:
        return

    from diffusers.utils import load_image

    image = load_image(str(args.image_path)).resize((args.width, args.height))
    identity = {
        "style": style,
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "image_size": list(image.size),
    }
    if args.resume_from is not None:
        saved = args.resume_from.parent / "auto_prompt.json"
        if not saved.is_file():
            raise ValueError(
                "Resume requires the saved auto_prompt.json or an explicit --prompt-path"
            )
        result = json.loads(saved.read_text(encoding="utf-8"))
        if any(result.get(key) != value for key, value in identity.items()):
            raise ValueError("Saved auto prompt does not match the input image and style")
    else:
        print(f"[worldcrafter] generating {args.prompt} prompt with {args.caption_model}")
        result = generate_caption(image, style, args.caption_model, args.device)
        result.update(identity)

    args.prompt = result["prompt"]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output_path.with_suffix(".prompt.txt")
    prompt_path.write_text(args.prompt + "\n", encoding="utf-8")
    metadata = json.dumps(result, indent=2) + "\n"
    args.output_path.with_suffix(".caption.json").write_text(metadata, encoding="utf-8")
    if args.state_output_dir is not None:
        args.state_output_dir.mkdir(parents=True, exist_ok=True)
        (args.state_output_dir / "auto_prompt.json").write_text(metadata, encoding="utf-8")
    print(f"[worldcrafter] prompt saved to {prompt_path}:\n{args.prompt}")
