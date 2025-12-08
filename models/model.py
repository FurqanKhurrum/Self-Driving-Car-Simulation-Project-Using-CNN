# scripts/model.py

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SHAPE = (66, 200, 3)

def nvidia_model():
    model = models.Sequential()
    model.add(layers.Input(shape=IMG_SHAPE))

    model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(1))  # steering

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse"
    )
    return model
