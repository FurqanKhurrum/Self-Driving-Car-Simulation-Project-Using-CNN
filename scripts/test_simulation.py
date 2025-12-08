# scripts/test_simulation.py

import argparse
import base64
import io
import os

import numpy as np
from PIL import Image
from flask import Flask
from flask_socketio import SocketIO, emit
import tensorflow as tf

from scripts.data_utils import preprocess_image  # reuse your preprocessing!


# ----------------- Model loading -----------------

def load_steering_model(model_path):
    print(f"[INFO] Loading model from: {model_path}")
    # Don't load the compiled training config (loss/metrics)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Optional: recompile for consistency (not strictly needed for predict)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse"
    )

    print("[INFO] Model loaded successfully.")
    return model



# ----------------- Flask / Socket.IO setup -----------------

app = Flask(__name__)
socketio = SocketIO(app)

model = None  # will be set in main()
MAX_SPEED = 25.0
MIN_SPEED = 10.0
speed_limit = MAX_SPEED


@socketio.on("connect")
def connect():
    print("[INFO] Simulator connected.")
    emit("manual", data={}, namespace='/', broadcast=True)


@socketio.on("telemetry")
def telemetry(data):
    """
    This is called every frame by the simulator.

    data contains:
    - steering_angle (current steering from sim)
    - throttle
    - speed
    - image (base64 string)
    """
    global speed_limit

    if data is None:
        return

    # 1) Read telemetry values (not all are needed)
    cur_steering = float(data["steering_angle"])
    cur_throttle = float(data["throttle"])
    cur_speed = float(data["speed"])

    # 2) Decode image from base64 → PIL → numpy array
    img_str = data["image"]
    img_bytes = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_bytes))
    image = np.asarray(image)  # RGB

    # 3) Preprocess image using SAME pipeline as training
    proc = preprocess_image(image)          # (66, 200, 3)
    proc = np.expand_dims(proc, axis=0)    # (1, 66, 200, 3)

    # 4) Predict steering angle
    steering_pred = float(model.predict(proc, verbose=0)[0][0])

    # 5) Simple speed control logic
    #    Slow down if turning sharply, speed up on straights
    if abs(steering_pred) > 0.3:
        speed_limit = MIN_SPEED
    else:
        speed_limit = MAX_SPEED

    if cur_speed > speed_limit:
        throttle = 0.0
    else:
        throttle = 0.3

    # 6) Send control back to simulator
    send_control(steering_pred, throttle)


def send_control(steering_angle, throttle):
    """
    Sends steering + throttle values to the simulator.
    They must be strings.
    """
    socketio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle),
        },
        skip_sid=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("models", "nvidia_model.h5"),
        help="Path to trained model file (.h5)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4567,
        help="Port to listen on (default: 4567)",
    )
    args = parser.parse_args()

    global model
    model = load_steering_model(args.model)

    print(f"[INFO] Listening on port {args.port}...")
    # host='0.0.0.0' allows external simulator; for local, this is fine too
    socketio.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
