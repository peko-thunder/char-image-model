import tensorflow as tf
import keras
from src.utils.config import DatasetConfig


def load_dataset(
    config: DatasetConfig,
) -> tuple[tf.data.Dataset | None, tf.data.Dataset | None]:
    train_ds = _load_train_ds(config)
    val_ds = _load_val_ds(config)

    return (train_ds, val_ds)


def _load_train_ds(config: DatasetConfig) -> tf.data.Dataset | None:
    print("Attempting to load training set...")
    try:
        return keras.utils.image_dataset_from_directory(
            config.dataset_dir,
            validation_split=config.validation_split,
            subset="training",
            seed=config.seed,
            image_size=config.image_size,
            batch_size=config.batch_size,
            label_mode="categorical",
        )
    except ValueError as e:
        print(f"Warning loading train: {e}")
        try:
            return keras.utils.image_dataset_from_directory(
                config.dataset_dir,
                validation_split=None,
                image_size=config.image_size,
                batch_size=config.batch_size,
                label_mode="categorical",
            )
        except Exception as e:
            print(f"Failed loading train: {e}")
            return None


def _load_val_ds(config: DatasetConfig) -> tf.data.Dataset | None:
    print("Attempting to load validation set...")
    try:
        return keras.utils.image_dataset_from_directory(
            config.dataset_dir,
            validation_split=config.validation_split,
            subset="validation",
            seed=config.seed,
            image_size=config.image_size,
            batch_size=config.batch_size,
            label_mode="categorical",
        )
    except Exception as e:
        print(f"Failed loading valid: {e}")
        return None
