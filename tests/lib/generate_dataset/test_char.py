from src.lib.generate_dataset.char import load_chars


def test_load_chars():
    chars = load_chars(
        [
            "chars/jis_x_0208_level1_kanji.txt",
            "chars/jis_x_0208_level2_kanji.txt",
        ]
    )
    assert len(chars) == 6355
