import os
from PIL import Image
from src.lib.generate_dataset.image import generate_font_char_image


def test_generate_font_char_image():
    font_path = "fonts/ipamjm00601/ipamjm.ttf"
    char = "﨑"
    output = "tests/dataset/test_output.jpg"

    try:
        generate_font_char_image(font_path, char, output, image_size=256)

        assert os.path.exists(output)
        with Image.open(output) as img:
            assert img.size == (256, 256)
    finally:
        if os.path.exists(output):
            os.remove(output)


def test_generate_font_char_image_custom_size():
    font_path = "fonts/ipamjm00601/ipamjm.ttf"
    char = "﨑"
    output = "tests/dataset/test_output_custom.jpg"

    try:
        generate_font_char_image(font_path, char, output, image_size=512, font_size=400)

        assert os.path.exists(output)
        with Image.open(output) as img:
            assert img.size == (512, 512)
    finally:
        if os.path.exists(output):
            os.remove(output)
