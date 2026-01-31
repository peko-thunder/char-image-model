import tensorflow as tf
import keras
import os


def create_vgg_model(input_shape, num_classes):
    """
    Creates a VGG16-based model for image classification.
    Uses Transfer Learning with ImageNet weights.
    """
    # Load VGG16 base model
    # include_top=False removes the final fully connected layers (the "head")
    base_model = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze the base model layers to prevent updating their weights during initial training
    base_model.trainable = False

    # Create the model architecture
    inputs = keras.Input(shape=input_shape)

    # Preprocess inputs to match VGG16 requirements (converts RGB to BGR, zero-centers)
    x = keras.applications.vgg16.preprocess_input(inputs)

    # Pass through the base model
    x = base_model(x, training=False)

    # Add custom top layers
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.models.Model(inputs, outputs, name="VGG16_Custom")
    return model


def main():
    # Parameters
    DATASET_DIR = "dataset"
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 5

    # Check dataset directory
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    print(f"TensorFlow Version: {tf.__version__}")
    print("Preparing dataset...")

    # Load Training Data
    val_ds = None
    try:
        train_ds = keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            label_mode="categorical",  # Returns one-hot encoded labels
        )

        print("Training set loaded successfully.")

        # Load Validation Data
        print("Attempting to load validation set...")
        try:
            val_ds = keras.utils.image_dataset_from_directory(
                DATASET_DIR,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE,
                label_mode="categorical",
            )
        except Exception as e:
            print(
                f"Warning: Validation split failed ({type(e).__name__}: {e}). Proceeding without validation data."
            )
            val_ds = None

    except ValueError as e:
        print(f"Initial loading failed: {e}")
        # If splitting failed entirely (e.g. extremely small dataset), try loading all as training
        print(
            "Warning: Dataset too small for splitting. Loading all data for training."
        )
        train_ds = keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=None,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )
        val_ds = None

    class_names = train_ds.class_names  # type: ignore
    num_classes = len(class_names)
    print(f"Detected classes: {class_names[:5]}, ...")

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)  # type: ignore
    if val_ds:
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)  # type: ignore

    # return None

    # Build Model
    print("Building VGG16 model...")
    model = create_vgg_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=num_classes
    )

    # Compile Model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    # Train Model
    print("Starting training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Save Model
    save_path = "vgg16_model.keras"
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
