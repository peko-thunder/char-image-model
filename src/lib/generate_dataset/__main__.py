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
        "fonts/BIZ_UDPGothic/BIZUDPGothic-Bold.ttf",  # ゴシック
        "fonts/BIZ_UDPGothic/BIZUDPGothic-Regular.ttf",  # ゴシック
        "fonts/ipamjm00601/ipamjm.ttf",  # 明朝
        "fonts/Noto_Sans_JP/static/NotoSansJP-Black.ttf",  # ゴシック
        "fonts/Noto_Sans_JP/static/NotoSansJP-Bold.ttf",  # ゴシック
        "fonts/Noto_Sans_JP/static/NotoSansJP-ExtraBold.ttf",  # ゴシック
        "fonts/Noto_Sans_JP/static/NotoSansJP-ExtraLight.ttf",  # ゴシック
        "fonts/Noto_Sans_JP/static/NotoSansJP-Light.ttf",  # ゴシック
        "fonts/Noto_Sans_JP/static/NotoSansJP-Medium.ttf",  # ゴシック
        "fonts/Noto_Sans_JP/static/NotoSansJP-Regular.ttf",  # ゴシック
        "fonts/Noto_Sans_JP/static/NotoSansJP-SemiBold.ttf",  # ゴシック
        "fonts/Noto_Sans_JP/static/NotoSansJP-Thin.ttf",  # ゴシック
        "fonts/Noto_Serif_JP/static/NotoSerifJP-Black.ttf",  # 明朝
        "fonts/Noto_Serif_JP/static/NotoSerifJP-Bold.ttf",  # 明朝
        "fonts/Noto_Serif_JP/static/NotoSerifJP-ExtraBold.ttf",  # 明朝
        "fonts/Noto_Serif_JP/static/NotoSerifJP-ExtraLight.ttf",  # 明朝
        "fonts/Noto_Serif_JP/static/NotoSerifJP-Light.ttf",  # 明朝
        "fonts/Noto_Serif_JP/static/NotoSerifJP-Medium.ttf",  # 明朝
        "fonts/Noto_Serif_JP/static/NotoSerifJP-Regular.ttf",  # 明朝
        "fonts/Noto_Serif_JP/static/NotoSerifJP-SemiBold.ttf",  # 明朝
        "fonts/Shippori_Antique/ShipporiAntique-Regular.ttf",  # ゴシック
        "fonts/Zen_Kaku_Gothic_Antique/ZenKakuGothicAntique-Black.ttf",  # ゴシック
        "fonts/Zen_Kaku_Gothic_Antique/ZenKakuGothicAntique-Bold.ttf",  # ゴシック
        "fonts/Zen_Kaku_Gothic_Antique/ZenKakuGothicAntique-Light.ttf",  # ゴシック
        "fonts/Zen_Kaku_Gothic_Antique/ZenKakuGothicAntique-Medium.ttf",  # ゴシック
        "fonts/Zen_Kaku_Gothic_Antique/ZenKakuGothicAntique-Regular.ttf",  # ゴシック
    ]

    chars = load_chars(
        [
            "chars/jis_x_0208_level3_kanji.txt",
            "chars/jis_x_0208_level4_kanji.txt",
        ]
    )

    IMAGE_SIZE = 50

    logs: list[str] = []

    for font_path, char in product(font_paths, chars):
        if not has_char(font_path, char):
            logs.append(f"{font_path} has not {char}")
            continue
        generate_image(char, font_path, dataset_path, IMAGE_SIZE)

    output_logs(logs)


def cleanup(dir: Path | str):
    dir_path = Path(dir)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir()


if __name__ == "__main__":
    run()
