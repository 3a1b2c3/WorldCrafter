"""Serving paths and the fixed I2V inference contract."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("WORLDCRAFTER_DEMO_DATA", ROOT.parent / "output/demo"))
MODEL_PATH = Path(
    os.environ.get("WORLDCRAFTER_DEMO_MODEL", ROOT.parent / "weights/WorldCrafter-Fast")
)
EXAMPLES = ROOT.parent / "test" / "I2V"
DEFAULT_PRESET = "01_socrates"
ROUTE = ("A",) * 5 + ("B",)


def presets():
    return [
        dict(
            id=case.name,
            name=case.name.split("_", 1)[-1].replace("_", " ").title(),
            image=f"/inputs/{case.name}/image.png",
            prompt=(case / "prompt.txt").read_text(encoding="utf-8").strip(),
        )
        for case in sorted(EXAMPLES.iterdir())
        if (case / "image.png").is_file() and (case / "prompt.txt").is_file()
    ]
