# scripts/test_simulation.py

print("Setting UP")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["EVENTLET_NO_GREENDNS"] = "yes"  # keep eventlet from messing with DNS

import argparse
import base64
from io import BytesIO

import eventlet
import numpy as np
import socketio
import cv2
from PIL import Image
from flask import Flask
from tensorflow.keras.models import load_model

# ----------------- Socket.IO + Flask setup -----------------

sio = socketio.Server(logger=False, engineio_logger=False)
app = Flask(__name__)  # '__main__'

# These will be set in main() from CLI args
maxSpeed = 10.0
STEERING_BIAS = 0.05
STEERING_SCALE = 1.0
SMOOTHING_ALPHA = 0.0  # 0 = no smoothing, 1 = full history (we'll clamp)
_prev_steering = 0.0   # for smoothing


def preProcessing(img):
    """
    EXACTLY the same as the function in data_utils.
    """
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


# ----------------- Event handlers -----------------

@sio.on("telemetry")
def telemetry(sid, data):
    global _prev_steering

    if data is None:
        return

    speed = float(data["speed"])

    # Decode simulator image
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)           # RGB
    image = preProcessing(image)
    image = np.array([image])           # shape (1, 66, 200, 3)

    # Predict steering from model
    raw_steering = float(model.predict(image, verbose=0)[0][0])

    # Optional scaling (shrink overly aggressive steering)
    scaled_steering = raw_steering * STEERING_SCALE

    # Apply small bias (to counter consistent drift)
    biased_steering = scaled_steering + STEERING_BIAS

    # Optional smoothing: exponential moving average with previous steering
    if SMOOTHING_ALPHA > 0.0:
        steering = (
            SMOOTHING_ALPHA * _prev_steering
            + (1.0 - SMOOTHING_ALPHA) * biased_steering
        )
    else:
        steering = biased_steering

    # Save for next frame
    _prev_steering = steering

    # Clamp steering to valid range
    steering = max(-1.0, min(1.0, steering))

    # Throttle logic from tutorial: slow down as speed approaches maxSpeed
    throttle = 1.0 - speed / maxSpeed

    print(
        f"raw={raw_steering:.4f}, scaled={scaled_steering:.4f}, "
        f"biased={biased_steering:.4f}, final={steering:.4f}, "
        f"throttle={throttle:.3f}, speed={speed:.2f}"
    )

    sendControl(steering, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("Connected:", sid)
    sendControl(0.0, 0.0)


def sendControl(steering, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering),
            "throttle": str(throttle),
        },
    )


# ----------------- Main -----------------

def main():
    global model, maxSpeed, STEERING_BIAS, STEERING_SCALE, SMOOTHING_ALPHA

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/nvidia_model.h5",
        help="Path to trained model (.h5)",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=10.0,
        help="Maximum speed used in throttle logic",
    )
    parser.add_argument(
        "--steering-bias",
        type=float,
        default=0.05,
        help="Constant bias added to predicted steering (to counter drift)",
    )
    parser.add_argument(
        "--steering-scale",
        type=float,
        default=1.0,
        help="Multiply predicted steering by this factor (e.g., 0.8 to calm it)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Smoothing factor in [0,1). 0 = no smoothing, "
             "0.5 = average between current and previous steering.",
    )

    args = parser.parse_args()

    maxSpeed = args.max_speed
    STEERING_BIAS = args.steering_bias
    STEERING_SCALE = args.steering_scale
    SMOOTHING_ALPHA = max(0.0, min(0.99, args.smooth))  # clamp to [0, 0.99]

    print(f"[INFO] Loading model from: {args.model}")
    model = load_model(args.model, compile=False)
    model.compile(optimizer="adam", loss="mse")
    print("[INFO] Model loaded.")

    print(
        f"[INFO] maxSpeed={maxSpeed}, "
        f"bias={STEERING_BIAS}, scale={STEERING_SCALE}, smooth={SMOOTHING_ALPHA}"
    )

    # Wrap Flask app with Socket.IO middleware
    wrapped_app = socketio.Middleware(sio, app)

    print("[INFO] Starting eventlet WSGI server on port 4567...")
    eventlet.wsgi.server(eventlet.listen(("", 4567)), wrapped_app)


if __name__ == "__main__":
    main()
