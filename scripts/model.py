# scripts/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# dataset = whole book
# batch = a single page
# batch generator = person handing you the next page
# epoch = reading the whole book once
def nvidia_model(input_shape=(66, 200, 3), learning_rate=1e-4):
    """
    NVIDIA-style end-to-end self-driving model.
    """
    model = Sequential()

    model.add(
        Conv2D(
            24,
            (5, 5),
            strides=(2, 2),
            activation="elu",
            input_shape=input_shape,
        )
    )
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))

    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")

    return model


if __name__ == "__main__":
    m = nvidia_model()
    m.summary()
