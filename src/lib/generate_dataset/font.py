from fontTools.ttLib import TTFont


def has_char(font_path: str, char: str):
    """
    指定したフォントが対応の文字データを持っているかを返す関数
    """
    font = TTFont(font_path)
    for table in font["cmap"].tables:
        if ord(char) in table.cmap:
            return True

    return False
