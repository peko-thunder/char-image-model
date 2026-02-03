import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402
from src.utils.config import load_workspace_config  # noqa: E402


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

    # 設定とCSV読み込み
    config_path = workspace_path / "config.json"
    csv_path = workspace_path / "validation_results.csv"

    if not config_path.exists():
        st.error(f"設定ファイルが見つかりません: {config_path}")
        return

    config = load_workspace_config(config_path)
    base_image_width = config.dataset.image_size[1]  # (H, W) の W

    df = pd.read_csv(csv_path)

    # 表示設定
    st.sidebar.header("表示設定")
    image_scale = st.sidebar.slider("画像スケール", 0.5, 5.0, 1.0, 0.5)
    image_width = int(base_image_width * image_scale)

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

    st.divider()

    # ページネーション
    items_per_page = st.sidebar.slider("表示件数", 10, 100, 20, 10)
    total_pages = (len(filtered_df) + items_per_page - 1) // items_per_page

    if total_pages > 0:
        page = st.sidebar.number_input("ページ", 1, total_pages, 1)
    else:
        page = 1

    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_df))
    page_df = filtered_df.iloc[start_idx:end_idx]

    st.caption(f"{start_idx + 1} - {end_idx} / {total} 件")

    # テーブルヘッダー
    header_cols = st.columns([2, 2, 1, 1, 1, 1, 1])
    header_cols[0].markdown("**File Name**")
    header_cols[1].markdown("**Source Image**")
    header_cols[2].markdown("**True Label**")
    header_cols[3].markdown("**Predicted Label**")
    header_cols[4].markdown("**Dataset Type**")
    header_cols[5].markdown("**Confidence**")
    header_cols[6].markdown("**Result**")

    st.divider()

    # テーブル行
    for _, row in page_df.iterrows():
        cols = st.columns([2, 2, 1, 1, 1, 1, 1])

        # File Name
        file_name = Path(row["Image Path"]).name
        cols[0].text(file_name)

        # Source Image
        source_path = PROJECT_ROOT / row["Image Path"]
        if source_path.exists():
            cols[1].image(source_path, width=image_width)
        else:
            cols[1].text("Not found")

        # True Label (ラベル + 文字)
        true_char = label_to_char(row["True Label"])
        cols[2].markdown(
            f'<span style="font-size:50px">{true_char}</span><br><code>{row["True Label"]}</code>',
            unsafe_allow_html=True,
        )

        # Predicted Label (ラベル + 文字)
        pred_char = label_to_char(row["Predicted Label"])
        cols[3].markdown(
            f'<span style="font-size:50px">{pred_char}</span><br><code>{row["Predicted Label"]}</code>',
            unsafe_allow_html=True,
        )

        # Dataset Type
        cols[4].text(row["Dataset Type"])

        # Confidence
        cols[5].text(f"{row['Confidence']:.4f}")

        # Result
        if row["Result"] == "CORRECT":
            cols[6].success(row["Result"])
        else:
            cols[6].error(row["Result"])

        st.divider()


if __name__ == "__main__":
    main()
