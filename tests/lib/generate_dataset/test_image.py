import os
from PIL import Image
from src.lib.generate_dataset.image import generate_image


def test_generate_image():
    char = "﨑"
    font_path = "fonts/ipamjm00601/ipamjm.ttf"
    output_path = "tests/dataset/"
    image_path = output_path + "FA11/ipamjm.jpg"

    try:
        generate_image(char, font_path, output_path, image_size=256)

        assert os.path.exists(image_path)
        with Image.open(image_path) as img:
            assert img.size == (256, 256)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


def test_generate_image_custom_size():
    char = "﨑"
    font_path = "fonts/ipamjm00601/ipamjm.ttf"
    output_path = "tests/dataset/"
    image_path = output_path + "FA11/ipamjm.jpg"

    try:
        generate_image(char, font_path, output_path, image_size=512, font_size=400)

        assert os.path.exists(image_path)
        with Image.open(image_path) as img:
            assert img.size == (512, 512)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
