# scripts/train_model.py

import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from data_utils import load_image_paths_and_steering, batch_generator
from model import nvidia_model


# ---- config ----
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "merged_driving_log_balanced.csv")

BATCH_SIZE = 64
EPOCHS = 30
INPUT_SHAPE = (66, 200, 3)

MODEL_DIR = "models"
BEST_MODEL_NAME = "nvidia_best.h5"
FINAL_MODEL_NAME = "nvidia_model.h5"


def main():
    # 1) Load image paths + steering angles
    image_paths, steerings = load_image_paths_and_steering(DATA_DIR, CSV_PATH)
    print(f"[INFO] Total samples after balancing: {len(image_paths)}")

    # 2) Train / validation split
    x_train, x_val, y_train, y_val = train_test_split(
        image_paths,
        steerings,
        test_size=0.2,
        random_state=10,
        shuffle=True,
    )
    print(f"[INFO] Training samples:   {len(x_train)}")
    print(f"[INFO] Validation samples: {len(x_val)}")

    # 3) Generators
    train_gen = batch_generator(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        is_training=True,
    )
    val_gen = batch_generator(
        x_val,
        y_val,
        batch_size=BATCH_SIZE,
        is_training=False,
    )

    # 4) Build model (smaller LR than before)
    model = nvidia_model(input_shape=INPUT_SHAPE, learning_rate=5e-5)
    model.summary()

    steps_per_epoch = len(x_train) // BATCH_SIZE
    val_steps = len(x_val) // BATCH_SIZE

    # 5) Callbacks: save best, stop early, reduce LR on plateau
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)

    callbacks = [
        ModelCheckpoint(
            best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 6) Train
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # 7) Save final model (after EarlyStopping restored best weights)
    final_model_path = os.path.join(MODEL_DIR, FINAL_MODEL_NAME)
    model.save(final_model_path)
    print(f"[INFO] Final model saved to: {final_model_path}")
    print(f"[INFO] Best model (by val_loss) saved to: {best_model_path}")

    # 8) Plot training vs validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
