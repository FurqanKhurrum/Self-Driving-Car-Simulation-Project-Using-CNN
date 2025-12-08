# scripts/data_utils.py

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
CSV_PATH = os.path.join("data", "driving_log.csv")

# Image shape expected by the Nvidia model
IMG_SHAPE = (66, 200, 3)

# Downsampling settings
ZERO_THRESH = 0.02          # |angle| < 0.02 → treat as "straight"
KEEP_PROB_STRAIGHT = 0.2    # keep only 20% of those straight samples


def load_driving_log(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["center", "left", "right",
                  "steering", "throttle", "brake", "speed"]
    # Use only center + steering
    samples = df[["center", "steering"]].values.tolist()
    return samples


def read_image(center_path):
    center_path = center_path.strip()

    # If CSV stores just the filename, join with IMG folder
    if not os.path.isabs(center_path):
        # handle paths like "IMG/xxx.jpg"
        if "IMG" in center_path:
            img_path = os.path.join("data", center_path)
        else:
            img_path = os.path.join("data", "IMG", center_path)
    else:
        img_path = center_path

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# ---------- Augmentation ----------

def random_flip(image, steering):
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        steering = -steering
    return image, steering


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    factor = 0.25 + np.random.rand()  # 0.25–1.25
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment_image(image, steering):
    # keep it simple for now
    if np.random.rand() < 0.5:
        image, steering = random_flip(image, steering)
    if np.random.rand() < 0.5:
        image = random_brightness(image)
    return image, steering


# ---------- Preprocessing ----------

def preprocess_image(image):
    # 1) Crop out sky + hood (tweak if needed)
    image = image[60:-25, :, :]  # from 160x320 → ~75x320

    # 2) Resize to (66, 200)
    image = cv2.resize(image, (200, 66))

    # 3) YUV color space (Nvidia)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # 4) Optional blur
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # 5) Normalize to [-0.5, 0.5]
    image = image.astype(np.float32)
    image = (image / 255.0) - 0.5

    return image


# ---------- Batch generator with downsampling ----------

def batch_generator(samples, batch_size=64, training=True):
    num_samples = len(samples)
    while True:
        np.random.shuffle(samples)

        images = []
        angles = []

        for center_path, steering in samples:
            steering = float(steering)

            # Downsample straight lines during TRAINING only
            if training and abs(steering) < ZERO_THRESH:
                if np.random.rand() > KEEP_PROB_STRAIGHT:
                    continue  # skip this sample

            img = read_image(center_path)

            if training:
                img, steering = augment_image(img, steering)

            img = preprocess_image(img)

            images.append(img)
            angles.append(steering)

            # When batch is full, yield it
            if len(images) == batch_size:
                yield np.array(images), np.array(angles)
                images, angles = [], []

        # If loop ends and there are leftovers, yield them too
        if images:
            yield np.array(images), np.array(angles)


def train_val_split(test_size=0.2):
    samples = load_driving_log(CSV_PATH)
    train_samples, val_samples = train_test_split(
        samples, test_size=test_size, shuffle=True, random_state=42
    )
    return train_samples, val_samples
