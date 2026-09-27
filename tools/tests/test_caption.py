import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from worldcrafter.caption import prepare_prompt
from worldcrafter.cli import parse_args


class CaptionTests(unittest.TestCase):
    def test_auto_requires_image_mode(self):
        for prompt in ("auto-first-person", "auto-third-person"):
            with self.assertRaisesRegex(ValueError, "require --mode i2v"):
                parse_args(["--mode", "t2v", "--prompt", prompt])

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
