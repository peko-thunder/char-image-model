from src.lib.generate_dataset.font import has_char


def test_has_char():
    font_path = "fonts/ipamjm00601/ipamjm.ttf"
    char = "﨑"
    assert has_char(font_path, char)
