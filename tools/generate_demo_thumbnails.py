"""Build lightweight demo previews from the I2V example images."""

from pathlib import Path

from PIL import Image, ImageOps


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "demo/static/thumbnails"
    target.mkdir(parents=True, exist_ok=True)
    original_bytes = preview_bytes = count = 0
    for source in sorted((root / "test/I2V").glob("*/image.png")):
        if not (source.parent / "prompt.txt").is_file():
            continue
        with Image.open(source) as image:
            preview = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGB"),
                (320, 192),
                method=Image.Resampling.LANCZOS,
            )
        output = target / f"{source.parent.name}.webp"
        preview.save(output, "WEBP", quality=78, method=6)
        original_bytes += source.stat().st_size
        preview_bytes += output.stat().st_size
        count += 1
    print(f"{count} previews: {original_bytes:,} -> {preview_bytes:,} bytes")


if __name__ == "__main__":
    main()
