import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from worldcrafter.caption import AUTO_PROMPTS, DEFAULT_CAPTION_MODEL, prepare_prompt
from worldcrafter.cli import parse_args


class CaptionTests(unittest.TestCase):
    def test_auto_requires_image_mode(self):
        for prompt in AUTO_PROMPTS:
            with self.assertRaisesRegex(ValueError, "require --mode i2v"):
                parse_args(["--mode", "t2v", "--prompt", prompt])

    def test_text_and_file_prompts(self):
        text = "A garden with flowers. " * 100
        self.assertEqual(parse_args(["--prompt", text]).prompt, text.strip())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene prompt.txt"
            path.write_text("A garden.\n", encoding="utf-8")
            for mode in ("i2v", "t2v"):
                args = parse_args(["--mode", mode, "--prompt", str(path)])
                self.assertEqual(args.prompt, "A garden.")
                self.assertFalse(hasattr(args, "prompt_path"))
            path.write_text(" \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                parse_args(["--prompt", str(path)])
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                parse_args(["--prompt", str(path)])

    def test_removed_argument_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--prompt-path", "prompt.txt"])

    def test_local_caption_weights_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory, patch("worldcrafter.cli.ROOT", Path(directory)):
            self.assertEqual(parse_args(["--prompt", "scene"]).caption_model, DEFAULT_CAPTION_MODEL)
            local = Path(directory) / "weights/Qwen3-VL-4B-Instruct"
            local.mkdir(parents=True)
            self.assertEqual(parse_args(["--prompt", "scene"]).caption_model, str(local))
            self.assertEqual(parse_args(["--caption-model", "chosen", "--prompt", "scene"]).caption_model, "chosen")

    def test_auto_uses_first_person_template(self):
        self.assertEqual(AUTO_PROMPTS["auto"], AUTO_PROMPTS["auto-first-person"])

    def test_ordinary_prompt_does_not_load_caption_model(self):
        args = parse_args(["--prompt", "An auto-first-person example in a classroom."])
        with patch("worldcrafter.caption.generate_caption", side_effect=AssertionError):
            prepare_prompt(args)
        self.assertEqual(args.prompt, "An auto-first-person example in a classroom.")

    def test_saved_prompt_resume_and_changed_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (640, 384), "green").save(image)
            argv = ["--image-path", str(image), "--prompt", "auto-first-person",
                    "--output-path", str(root / "video.mp4"),
                    "--state-output-dir", str(root / "states")]
            args = parse_args(argv)
            with patch("worldcrafter.caption.generate_caption", return_value={
                "prompt": "A green garden.", "style": "first_person",
            }) as generate:
                prepare_prompt(args)
                generate.assert_called_once()
            self.assertEqual(args.prompt, "A green garden.")
            self.assertEqual((root / "video.prompt.txt").read_text(), "A green garden.\n")
            saved = json.loads((root / "states/auto_prompt.json").read_text())
            self.assertEqual(saved["image_size"], [640, 384])
            resume = argv + ["--resume-from", str(root / "states/chunk.pt"),
                             "--chunk-output-dir", str(root / "chunks")]
            with patch("worldcrafter.caption.generate_caption", side_effect=AssertionError):
                resumed = parse_args(resume)
                prepare_prompt(resumed)
                self.assertEqual(resumed.prompt, args.prompt)
                Image.new("RGB", (640, 384), "red").save(image)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    prepare_prompt(parse_args(resume))

    def test_caption_failure_is_not_used_as_video_prompt(self):
        args = parse_args(["--prompt", "auto-third-person"])
        with patch("worldcrafter.caption.generate_caption", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                prepare_prompt(args)


if __name__ == "__main__":
    unittest.main()
