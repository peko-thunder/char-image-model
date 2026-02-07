import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import keras
from pathlib import Path
from src.utils.dataset import load_dataset
from src.utils.config import load_workspace_config


def _train(
    dataset: tuple[tf.data.Dataset, tf.data.Dataset | None],
    image_size: tuple[int, int],
    num_classes: int,
    epochs: int,
    dropout_rate: float,
) -> keras.Model:
    (train_ds, val_ds) = dataset

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    if val_ds:
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build Model
    print("Building EfficientNetB0 model...")
    model = _create_model(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )

    # Compile Model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train Model
    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    return model


def _create_model(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout_rate: float,
    use_augmentation: bool = True,
) -> keras.Model:
    """
    Creates an EfficientNetB0-based model for image classification.
    Uses Transfer Learning with ImageNet weights.
    Optimized for small images (50x50).
    """
    # Load EfficientNetB0 base model
    base_model = keras.applications.EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze the base model layers
    base_model.trainable = True

    # Data augmentation (only active during training)
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomZoom(0.05),  # ±5%
        ],
        name="data_augmentation",
    )

    # Create the model architecture
    inputs = keras.Input(shape=input_shape)

    # Apply data augmentation (automatically disabled during inference)
    x = data_augmentation(inputs) if use_augmentation else inputs

    # EfficientNet has built-in preprocessing
    x = keras.applications.efficientnet.preprocess_input(x)

    # Pass through the base model
    x = base_model(x, training=False)

    # Add custom top layers
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.models.Model(inputs, outputs, name="EfficientNetB0_Custom")
    return model


def run(workspace: Path | str):
    workspace_path = Path(workspace)
    config_path = workspace_path / "config.json"

    # Load config
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.")
        return
    print(f"Loading config from {config_path}...")
    config = load_workspace_config(config_path)

    # Check dataset directory
    if not os.path.exists(config.dataset.dataset_dir):
        print(f"Error: Dataset directory '{config.dataset.dataset_dir}' not found.")
        return

    print(f"TensorFlow Version: {tf.__version__}")
    print("Preparing dataset...")

    (train_ds, val_ds) = load_dataset(config.dataset)
    if not train_ds:
        return

    class_names = train_ds.class_names  # type: ignore
    num_classes = len(class_names)
    print(f"Detected classes: {class_names[:5]} and others...")

    model = _train(
        dataset=(train_ds, val_ds),
        image_size=config.dataset.image_size,
        num_classes=num_classes,
        epochs=config.training.epochs,
        dropout_rate=config.training.dropout_rate,
    )

    # Save Model
    output_path = workspace_path / "model.keras"
    model.save(output_path)
    print(f"Model saved to {str(output_path)}")


if __name__ == "__main__":
    run(Path(__file__).parent)
