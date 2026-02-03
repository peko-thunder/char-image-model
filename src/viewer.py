import base64
from pathlib import Path

import streamlit as st
import pandas as pd

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def image_to_base64(path: str) -> str | None:
    """画像をbase64エンコードしたdata URIに変換"""
    p = Path(path)
    if not p.exists():
        return None
    suffix = p.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }
    mime = mime_types.get(suffix, "image/png")
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def get_workspaces() -> list[str]:
    """ワークスペース一覧を取得"""
    base_path = PROJECT_ROOT / "workspace"
    if not base_path.exists():
        return []
    return sorted(
        [
            d.name
            for d in base_path.iterdir()
            if d.is_dir() and (d / "validation_results.csv").exists()
        ]
    )


def label_to_char(label: str) -> str:
    """ユニコードコードポイント(16進数)から文字に変換"""
    try:
        return chr(int(label, 16))
    except (ValueError, OverflowError):
        return "?"


def main():
    st.set_page_config(
        page_title="Validation Results Viewer",
        layout="wide",
    )

    # 余白を調整するCSS
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            display: none;
        }
        .stMainBlockContainer {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Validation Results Viewer")

    # ワークスペース選択
    workspaces = get_workspaces()
    if not workspaces:
        st.error(
            "ワークスペースが見つかりません。validation_results.csvを含むワークスペースを作成してください。"
        )
        return

    selected_workspace = st.selectbox("ワークスペース", workspaces)
    workspace_path = PROJECT_ROOT / "workspace" / selected_workspace

    # CSV読み込み
    csv_path = workspace_path / "validation_results.csv"
    df = pd.read_csv(csv_path)

    # フィルタリング
    st.sidebar.header("フィルター")

    # 文字検索フィルター
    search_char = st.sidebar.text_input("文字検索", placeholder="例: あ")
    search_labels = []
    if search_char:
        search_labels = [format(ord(c), "04X") for c in search_char]

    # Result フィルター
    results = ["ALL"] + sorted(df["Result"].unique().tolist())
    selected_result = st.sidebar.selectbox("Result", results)

    # Dataset Type フィルター
    dataset_types = ["ALL"] + sorted(df["Dataset Type"].unique().tolist())
    selected_dataset_type = st.sidebar.selectbox("Dataset Type", dataset_types)

    # True Label フィルター
    true_labels = ["ALL"] + sorted(df["True Label"].unique().tolist())
    selected_true_label = st.sidebar.selectbox("True Label", true_labels)

    # Predicted Label フィルター
    pred_labels = ["ALL"] + sorted(df["Predicted Label"].unique().tolist())
    selected_pred_label = st.sidebar.selectbox("Predicted Label", pred_labels)

    # フィルター適用
    filtered_df = df.copy()
    if search_labels:
        filtered_df = filtered_df[
            filtered_df["True Label"].isin(search_labels)
            | filtered_df["Predicted Label"].isin(search_labels)
        ]
    if selected_result != "ALL":
        filtered_df = filtered_df[filtered_df["Result"] == selected_result]
    if selected_dataset_type != "ALL":
        filtered_df = filtered_df[filtered_df["Dataset Type"] == selected_dataset_type]
    if selected_true_label != "ALL":
        filtered_df = filtered_df[filtered_df["True Label"] == selected_true_label]
    if selected_pred_label != "ALL":
        filtered_df = filtered_df[filtered_df["Predicted Label"] == selected_pred_label]

    # 統計情報
    total = len(filtered_df)
    correct = len(filtered_df[filtered_df["Result"] == "CORRECT"])
    wrong = len(filtered_df[filtered_df["Result"] == "WRONG"])
    accuracy = correct / total * 100 if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", total)
    col2.metric("Correct", correct)
    col3.metric("Wrong", wrong)
    col4.metric("Accuracy", f"{accuracy:.2f}%")

    # ページネーション
    items_per_page = st.sidebar.slider("表示件数", 10, 100, 20, 10)
    total_pages = max(1, (total + items_per_page - 1) // items_per_page)
    page = st.sidebar.number_input("ページ", 1, total_pages, 1)

    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total)

    st.caption(f"{start_idx + 1} - {end_idx} / {total} 件")

    # 表示するページ分のみ処理
    page_df = filtered_df.iloc[start_idx:end_idx].copy()

    # ファイル名を抽出
    page_df["File Name"] = page_df["Image Path"].apply(lambda x: Path(x).name)

    # 画像をbase64エンコード（表示分のみ）
    page_df["Image"] = page_df["Image Path"].apply(
        lambda x: image_to_base64(str(PROJECT_ROOT / x))
    )

    # ラベルに文字を追加
    page_df["True"] = page_df["True Label"].apply(lambda x: f"{label_to_char(x)} ({x})")
    page_df["Predicted"] = page_df["Predicted Label"].apply(
        lambda x: f"{label_to_char(x)} ({x})"
    )

    # 表示用カラム選択
    display_df = page_df[
        [
            "File Name",
            "Image",
            "True",
            "Predicted",
            "Dataset Type",
            "Confidence",
            "Result",
        ]
    ]

    # データフレーム表示
    st.dataframe(
        display_df,
        column_config={
            "File Name": st.column_config.TextColumn("File Name", width="medium"),
            "Image": st.column_config.ImageColumn("Source Image", width="medium"),
            "True": st.column_config.TextColumn("True Label", width="small"),
            "Predicted": st.column_config.TextColumn("Predicted Label", width="small"),
            "Dataset Type": st.column_config.TextColumn("Dataset Type", width="small"),
            "Confidence": st.column_config.NumberColumn(
                "Confidence", format="%.4f", width="small"
            ),
            "Result": st.column_config.TextColumn("Result", width="small"),
        },
        hide_index=True,
        width="stretch",
    )


if __name__ == "__main__":
    main()
