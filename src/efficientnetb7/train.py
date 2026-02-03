import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import keras
from pathlib import Path
from src.utils.dataset import load_dataset
from src.utils.config import DatasetConfig, TrainingConfig, save_config


def _train(
    dataset: tuple[tf.data.Dataset, tf.data.Dataset | None],
    config: DatasetConfig,
    epochs: int,
    dropout_rate: float,
) -> keras.Model:
    (train_ds, val_ds) = dataset

    class_names = train_ds.class_names  # type: ignore
    num_classes = len(class_names)
    print(f"Detected classes: {class_names[:5]} and others...")

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    if val_ds:
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build Model
    print("Building EfficientNetB7 model...")
    model = _create_model(
        input_shape=(config.image_size[0], config.image_size[1], 3),
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
    Creates an EfficientNetB7-based model for image classification.
    Uses Transfer Learning with ImageNet weights.
    """
    # Load EfficientNetB7 base model
    base_model = keras.applications.EfficientNetB7(
        # weights="imagenet",
        weights=None,
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze the base model layers
    # base_model.trainable = False
    base_model.trainable = True

    # Data augmentation (only active during training)
    data_augmentation = keras.Sequential(
        [
            # keras.layers.RandomRotation(0.05),  # ±18 degrees
            keras.layers.RandomZoom(0.05),  # ±5%
            # keras.layers.RandomTranslation(0.05, 0.05),  # ±5% shift
        ],
        name="data_augmentation",
    )

    # Create the model architecture
    inputs = keras.Input(shape=input_shape)

    # Apply data augmentation (automatically disabled during inference)
    x = data_augmentation(inputs) if use_augmentation else inputs

    # EfficientNet has built-in preprocessing, but we apply it explicitly
    x = keras.applications.efficientnet.preprocess_input(x)

    # Pass through the base model
    x = base_model(x, training=False)

    # Add custom top layers
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.models.Model(inputs, outputs, name="EfficientNetB7_Custom")
    return model


def run():
    DATASET_DIR = "dataset/default"
    IMAGE_SIZE = (50, 50)
    BATCH_SIZE = 64  # B7 is large, smaller batch size recommended
    VALIDATION_SPLIT = 0.2
    SEED = 123
    EPOCHS = 100
    DROPOUT_RATE = 0.5
    OUTPUT_DIR = "models/efficientnetb7/default"

    # Check dataset directory
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    print(f"TensorFlow Version: {tf.__version__}")
    print("Preparing dataset...")

    config = DatasetConfig(
        dataset_dir=DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
    )

    (train_ds, val_ds) = load_dataset(config)
    if not train_ds:
        return

    model = _train(
        dataset=(train_ds, val_ds),
        config=config,
        epochs=EPOCHS,
        dropout_rate=DROPOUT_RATE,
    )

    # Save Model
    output_path = Path(OUTPUT_DIR) / "model.keras"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {str(output_path)}")

    # Save Configs
    dataset_config_path = output_path.parent / "dataset_config.json"
    save_config(config, dataset_config_path)
    print(f"Dataset config saved to {str(dataset_config_path)}")

    training_config = TrainingConfig(epochs=EPOCHS, dropout_rate=DROPOUT_RATE)
    training_config_path = output_path.parent / "training_config.json"
    save_config(training_config, training_config_path)
    print(f"Training config saved to {str(training_config_path)}")


if __name__ == "__main__":
    run()
