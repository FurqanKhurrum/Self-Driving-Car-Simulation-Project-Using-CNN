# scripts/test_simulation.py

print("Setting UP")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["EVENTLET_NO_GREENDNS"] = "yes"

import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
from scripts.data_utils import preprocess_image

# ----- Socket.IO + Flask -----

# logger=True, engineio_logger=True to see handshake issues
sio = socketio.Server()
app = Flask(__name__)  # '__main__'

maxSpeed = 10

@sio.on("telemetry")
def telemetry(sid, data):
    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)           # RGB
    image = preprocess_image(image)     # your training pipeline
    image = np.array([image])           # shape (1, 66, 200, 3) or whatever you used
    steering = float(model.predict(image, verbose=0)[0][0])
    steering *= 0.7                      # scale down steering by 30%
    steering = max(-1.0, min(1.0, steering))

    throttle = 1.0 - speed / maxSpeed
    print(f"steer={steering:.4f}, throttle={throttle:.3f}, speed={speed:.2f}")
    sendControl(steering, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("Connected:", sid)
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering),
            "throttle": str(throttle),
        },
    )


if __name__ == "__main__":
    model = load_model("models/nvidia_model.h5", compile=False)
    model.compile(optimizer="adam", loss="mse")

    app = socketio.Middleware(sio, app)
    print("[INFO] Starting eventlet WSGI server on port 4567...")
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)

