import shutil
from pathlib import Path


def copy_dataset(
    src_dir: Path | str,
    dst_dir: Path | str,
    file_names: list[str],
) -> dict[str, int]:
    """特定のファイル名だけを同じ構造で別フォルダに配置

    Args:
        src_dir: 元のデータセットディレクトリ
        dst_dir: 出力先ディレクトリ
        file_names: コピー対象のファイル名リスト (例: ["ipamjm.jpg"])

    Returns:
        統計情報 {"copied": int, "skipped": int, "classes": int}
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    stats = {"copied": 0, "skipped": 0, "classes": 0}

    for class_dir in sorted(src_path.iterdir()):
        if not class_dir.is_dir():
            continue

        dst_class_dir = dst_path / class_dir.name
        class_has_files = False

        for file_name in file_names:
            src_file = class_dir / file_name
            if src_file.exists():
                dst_class_dir.mkdir(parents=True, exist_ok=True)
                dst_file = dst_class_dir / file_name

                if dst_file.exists():
                    stats["skipped"] += 1
                    continue

                shutil.copy2(src_file, dst_file)

                stats["copied"] += 1
                class_has_files = True

        if class_has_files:
            stats["classes"] += 1

    return stats


if __name__ == "__main__":
    stats = copy_dataset(
        src_dir="dataset/20260207_all_50",
        dst_dir="dataset/20260207_gothic_50",
        file_names=[
            "BIZUDPGothic-Bold.jpg",  # ゴシック
            "BIZUDPGothic-Regular.jpg",  # ゴシック
            # "ipamjm.jpg",  # 明朝
            "NotoSansJP-Black.jpg",  # ゴシック
            "NotoSansJP-Bold.jpg",  # ゴシック
            "NotoSansJP-ExtraBold.jpg",  # ゴシック
            "NotoSansJP-ExtraLight.jpg",  # ゴシック
            "NotoSansJP-Light.jpg",  # ゴシック
            "NotoSansJP-Medium.jpg",  # ゴシック
            "NotoSansJP-Regular.jpg",  # ゴシック
            "NotoSansJP-SemiBold.jpg",  # ゴシック
            "NotoSansJP-Thin.jpg",  # ゴシック
            # "NotoSerifJP-Black.jpg",  # 明朝
            # "NotoSerifJP-Bold.jpg",  # 明朝
            # "NotoSerifJP-ExtraBold.jpg",  # 明朝
            # "NotoSerifJP-ExtraLight.jpg",  # 明朝
            # "NotoSerifJP-Light.jpg",  # 明朝
            # "NotoSerifJP-Medium.jpg",  # 明朝
            # "NotoSerifJP-Regular.jpg",  # 明朝
            # "NotoSerifJP-SemiBold.jpg",  # 明朝
            "ShipporiAntique-Regular.jpg",  # ゴシック
            "ZenKakuGothicAntique-Black.jpg",  # ゴシック
            "ZenKakuGothicAntique-Bold.jpg",  # ゴシック
            "ZenKakuGothicAntique-Light.jpg",  # ゴシック
            "ZenKakuGothicAntique-Medium.jpg",  # ゴシック
            "ZenKakuGothicAntique-Regular.jpg",  # ゴシック
        ],
    )

    print(f"Classes: {stats['classes']}")
    print(f"Copied: {stats['copied']}")
    print(f"Skipped: {stats['skipped']}")
