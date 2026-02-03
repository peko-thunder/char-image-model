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


def save_config(config: DatasetConfig | TrainingConfig, path: Path | str) -> None:
    """ConfigをJSONファイルとして保存"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)


def load_dataset_config(path: Path | str) -> DatasetConfig:
    """JSONファイルからDatasetConfigを読み込み"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetConfig(**data)


def load_training_config(path: Path | str) -> TrainingConfig:
    """JSONファイルからTrainingConfigを読み込み"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainingConfig(**data)
