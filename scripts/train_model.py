# scripts/train_model.py

import os
from scripts.data_utils import train_val_split, batch_generator
from models.model import nvidia_model


from scripts.data_utils import train_val_split, batch_generator
import tensorflow as tf

MODEL_PATH = os.path.join("models", "nvidia_model.h5")

def main():
    os.makedirs("models", exist_ok=True)

    train_samples, val_samples = train_val_split(test_size=0.2)

    batch_size = 64
    train_gen = batch_generator(train_samples, batch_size=batch_size, training=True)
    val_gen   = batch_generator(val_samples,   batch_size=batch_size, training=False)

    model = nvidia_model()
    model.summary()

    steps_per_epoch = max(1, len(train_samples) // batch_size)
    val_steps       = max(1, len(val_samples) // batch_size)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=10,
        callbacks=[checkpoint]
    )

if __name__ == "__main__":
    main()
