import keras
import numpy as np
import os
import csv


def main():
    # Configuration matches main.py
    MODEL_PATH = "vgg16_model.keras"
    DATASET_DIR = "dataset"
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    OUTPUT_CSV = "validation_results.csv"

    # 1. Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please run 'python3 src/main.py' to train and save the model first.")
        return

    # 2. Check dataset
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading validation dataset...")
    try:
        # ensuring the same split as training by using the same seed
        val_ds = keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
        )
    except ValueError as e:
        print(f"Error loading validation dataset: {e}")
        return

    class_names = val_ds.class_names  # type: ignore
    print(f"Target Classes: {class_names[:5]}, ...")

    # 3. Evaluate
    print("\n--- Model Evaluation ---")
    loss, accuracy = model.evaluate(val_ds)  # type: ignore
    print(f"Validation Loss:     {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # 4. Export All Predictions to CSV
    print(f"\n--- Exporting Predictions to {OUTPUT_CSV} ---")

    try:
        with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["True Label", "Predicted Label", "Confidence", "Result"])

            total_batches = len(val_ds)
            print(f"Processing {total_batches} batches...")

            for i, (images, labels) in enumerate(val_ds.as_numpy_iterator()):  # type: ignore
                # Predict batch
                predictions = model.predict(images, verbose=0)  # type: ignore

                # Convert one-hot encoded labels back to index
                pred_indices = np.argmax(predictions, axis=1)
                true_indices = np.argmax(labels, axis=1)

                rows_to_write = []
                for j in range(len(images)):
                    true_label = class_names[true_indices[j]]
                    pred_label = class_names[pred_indices[j]]
                    confidence = np.max(predictions[j])
                    result = "CORRECT" if true_label == pred_label else "WRONG"

                    rows_to_write.append(
                        [true_label, pred_label, f"{confidence:.4f}", result]
                    )

                writer.writerows(rows_to_write)

                # Simple progress indicator
                if (i + 1) % 10 == 0:
                    print(f"Processed batch {i + 1}/{total_batches}")

        print(f"Done! Results saved to '{OUTPUT_CSV}'.")

    except Exception as e:
        print(f"Error writing to CSV: {e}")


if __name__ == "__main__":
    main()
