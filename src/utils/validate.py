import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import argparse
import tensorflow as tf
import keras
import numpy as np
import csv
from typing import Literal, Sequence, TypedDict
from pathlib import Path
from src.utils.dataset import load_dataset
from src.utils.config import load_workspace_config


class PredictionResult(TypedDict):
    image_path: str
    dataset_type: Literal["train", "val"]
    true_label: str
    pred_label: str
    confidence: float


def collect_predictions_from_files(
    model: keras.Model,
    dataset: tf.data.Dataset,
    dataset_type: Literal["train", "val"],
    class_names: Sequence[str],
    image_size: tuple[int, int],
    batch_size: int = 32,
) -> list[PredictionResult]:
    """画像ファイルから直接予測結果を収集する（バッチ処理）"""

    file_paths = dataset.file_paths
    loss, accuracy = model.evaluate(dataset)
    print(f"Loss:     {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    results = []
    total_files = len(file_paths)
    total_batches = (total_files + batch_size - 1) // batch_size
    print(
        f"Processing {dataset_type} dataset: {total_files} files in {total_batches} batches..."
    )

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_paths = file_paths[start_idx:end_idx]

        # バッチの画像を読み込み
        batch_images = []
        for file_path in batch_paths:
            img = keras.utils.load_img(file_path, target_size=image_size)
            img_array = keras.utils.img_to_array(img)
            batch_images.append(img_array)

        batch_array = np.array(batch_images)
        predictions = model.predict(batch_array, verbose=0)

        # 結果を収集
        for i, file_path in enumerate(batch_paths):
            pred_idx = np.argmax(predictions[i])
            confidence = float(np.max(predictions[i]))
            true_label = Path(file_path).parent.name
            pred_label = class_names[pred_idx]

            results.append(
                {
                    "image_path": Path(file_path).as_posix(),
                    "dataset_type": dataset_type,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                }
            )

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed batch {batch_idx + 1}/{total_batches}")

    return results


def export_predictions_to_csv(
    predictions: list[PredictionResult],
    output: Path | str,
) -> None:
    """予測結果をCSVファイルに出力する（画像パスで昇順ソート）"""
    print(f"\n--- Exporting Predictions to {str(output)} ---")

    sorted_predictions = sorted(predictions, key=lambda x: x["image_path"])

    with open(str(output), mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Image Path",
                "Dataset Type",
                "True Label",
                "Predicted Label",
                "Confidence",
                "Result",
            ]
        )

        for pred in sorted_predictions:
            result = "CORRECT" if pred["true_label"] == pred["pred_label"] else "WRONG"
            writer.writerow(
                [
                    pred["image_path"],
                    pred["dataset_type"],
                    pred["true_label"],
                    pred["pred_label"],
                    f"{pred['confidence']:.4f}",
                    result,
                ]
            )

    print(f"Done! Results saved to '{str(output)}'.")


def validate(workspace: str, output_path: str | None = None) -> None:
    """モデルを検証し、結果をCSVに出力する

    Args:
        workspace: 個別ワークスペースのパス（例: workspace/exp001）
        output_path: 出力CSVのパス（省略時は指定ワークスペース内のvalidation_results.csv）
    """
    workspace_path = Path(workspace)
    model_path = workspace_path / "model.keras"
    config_path = workspace_path / "config.json"

    if output_path is None:
        output_path = str(workspace_path / "validation_results.csv")

    # 1. モデルの読み込み
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        return
    print(f"Loading model from {model_path}...")
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. 設定の読み込み
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.")
        return
    print(f"Loading config from {config_path}...")
    config = load_workspace_config(config_path)

    # 3. データセットの読み込み
    if not os.path.exists(config.dataset.dataset_dir):
        print(f"Error: Dataset directory '{config.dataset.dataset_dir}' not found.")
        return
    (train_ds, val_ds) = load_dataset(config.dataset)
    if not train_ds:
        return

    class_names = train_ds.class_names  # type: ignore
    print(f"Target Classes: {class_names[:5]} and others...")

    # 4. 予測結果の収集（バッチ処理）
    all_predictions = []

    try:
        train_predictions = collect_predictions_from_files(
            model,
            train_ds,
            "train",
            class_names,
            config.dataset.image_size,
            config.dataset.batch_size,
        )
        all_predictions.extend(train_predictions)
    except Exception as e:
        print(f"Error collecting training predictions: {e}")

    if val_ds:
        try:
            val_predictions = collect_predictions_from_files(
                model,
                val_ds,
                "val",
                class_names,
                config.dataset.image_size,
                config.dataset.batch_size,
            )
            all_predictions.extend(val_predictions)
        except Exception as e:
            print(f"Error collecting validation predictions: {e}")

    # 5. CSVに出力（画像パスでソート）
    try:
        export_predictions_to_csv(all_predictions, output_path)
    except Exception as e:
        print(f"Error exporting CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description="モデルの検証を実行")
    parser.add_argument(
        "workspace",
        type=str,
        help="個別ワークスペースのパス（例: workspace/exp001）",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="出力CSVのパス（省略時は指定ワークスペース内のvalidation_results.csv）",
    )
    args = parser.parse_args()

    validate(args.workspace, args.output)


if __name__ == "__main__":
    main()
