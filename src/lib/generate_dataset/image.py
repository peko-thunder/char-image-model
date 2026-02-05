from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def generate_image(
    char: str,
    font_path: Path | str,
    output_path: Path | str,
    image_size: int = 256,
    font_size: int | None = None,
) -> None:
    image_path = resolve_image_path(char, font_path, output_path)
    if font_size is None:
        font_size = int(image_size * 0.8)
    font = ImageFont.truetype(font_path, size=font_size)
    image = Image.new("RGB", (image_size, image_size), color="white")
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (image_size - text_width) / 2 - bbox[0]
    y = (image_size - text_height) / 2 - bbox[1]

    draw.text((x, y), char, font=font, fill="black")
    image.save(image_path)


def resolve_image_path(
    char: str,
    font_path: Path | str,
    output_path: Path | str,
) -> Path:
    font_name = Path(font_path).stem
    unicode_hex = format(ord(char), "04X")
    output_dir = Path(output_path) / unicode_hex
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{font_name}.jpg"
