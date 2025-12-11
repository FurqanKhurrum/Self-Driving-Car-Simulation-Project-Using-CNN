# scripts/test_simulation.py

print("Setting UP")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["EVENTLET_NO_GREENDNS"] = "yes"  # keep eventlet from messing with DNS

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

maxSpeed = 10.0
STEERING_BIAS = 0.05

def preProcessing(img):
    """
    EXACTLY the same as your teacher's version.
    If this is how you trained, this is what you should use here.
    """
    img = img[60:135, :, :]                     # crop sky/hood
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


# ----------------- Event handlers -----------------

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])

    # Decode simulator image
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)           # RGB
    image = preProcessing(image)
    image = np.array([image])           # shape (1, 66, 200, 3)

    # Predict steering from model
    raw_steering = float(model.predict(image, verbose=0)[0][0])

    # Apply small bias to counter left drift
    steering = raw_steering + STEERING_BIAS
    # Clamp steering to valid range
    steering = max(-1.0, min(1.0, steering))

    # Throttle logic from teacher
    throttle = 1.0 - speed / maxSpeed

    print(
        f"raw={raw_steering:.4f}, biased={steering:.4f}, "
        f"throttle={throttle:.3f}, speed={speed:.2f}"
    )

    sendControl(steering, throttle)



@sio.on('connect')
def connect(sid, environ):
    print('Connected:', sid)
    sendControl(0.0, 0.0)


def sendControl(steering, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': str(steering),
            'throttle': str(throttle),
        }
    )


# ----------------- Main -----------------

if __name__ == '__main__':
    # Load the trained model
    model = load_model('models/nvidia_model.h5', compile=False)
    # Optional: compile for consistency (not required for predict)
    model.compile(optimizer='adam', loss='mse')

    # Wrap Flask app with Socket.IO middleware
    app = socketio.Middleware(sio, app)

    print("[INFO] Starting eventlet WSGI server on port 4567...")
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
