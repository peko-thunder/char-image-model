import json
from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class DatasetConfig:
    dataset_dir: str
    image_size: tuple[int, int]
    batch_size: int
    validation_split: float | None = None
    seed: Any | None = None


@dataclass
class TrainingConfig:
    epochs: int
    dropout_rate: float


@dataclass
class WorkspaceConfig:
    dataset: DatasetConfig
    training: TrainingConfig


def save_workspace_config(config: WorkspaceConfig, path: Path | str) -> None:
    """WorkspaceConfigをJSONファイルとして保存"""
    data = {
        "dataset": asdict(config.dataset),
        "training": asdict(config.training),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_workspace_config(path: Path | str) -> WorkspaceConfig:
    """JSONファイルからWorkspaceConfigを読み込み"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_data = data["dataset"]
    # image_size をタプルに変換
    dataset_data["image_size"] = tuple(dataset_data["image_size"])

    return WorkspaceConfig(
        dataset=DatasetConfig(**dataset_data),
        training=TrainingConfig(**data["training"]),
    )
