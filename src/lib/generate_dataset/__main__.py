import shutil
from itertools import product
from pathlib import Path
from .char import load_chars
from .font import has_char
from .image import generate_image
from .log import output_logs


def run(dataset_path: Path = Path("dataset/tmp")):
    cleanup(dataset_path)

    font_paths = [
        "fonts/ipamjm00601/ipamjm.ttf",
        "fonts/Noto_Sans_JP/static/NotoSansJP-Light.ttf",
        "fonts/Noto_Sans_JP/static/NotoSansJP-Medium.ttf",
        "fonts/Noto_Sans_JP/static/NotoSansJP-ExtraBold.ttf",
        "fonts/Noto_Serif_JP/static/NotoSerifJP-Light.ttf",
        "fonts/Noto_Serif_JP/static/NotoSerifJP-Medium.ttf",
        "fonts/Noto_Serif_JP/static/NotoSerifJP-ExtraBold.ttf",
    ]
    chars = load_chars(
        [
            "chars/jis_x_0208_level3_kanji.txt",
            "chars/jis_x_0208_level4_kanji.txt",
        ]
    )
    logs: list[str] = []

    for font_path, char in product(font_paths, chars):
        if not has_char(font_path, char):
            logs.append(f"{font_path} has not {char}")
            continue
        generate_image(char, font_path, dataset_path, 50)

    output_logs(logs)


def cleanup(dir: Path | str):
    dir_path = Path(dir)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir()


if __name__ == "__main__":
    run()
