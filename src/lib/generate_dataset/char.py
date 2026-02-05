def load_chars(file_paths: list[str]) -> list[str]:
    """
    改行区切りのテキストファイルを読み込み、全ての文字をリストで返す関数
    """
    chars: list[str] = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chars.append(line)

    return chars
