# scripts/data_utils.py
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def balance_steering_data(df, n_bins=31, samples_per_bin=700, display=True):
    """
    Downsample over-represented steering angles so that each histogram bin
    has at most `samples_per_bin` samples.

    df: pandas DataFrame with a 'steering' column.
    """

    hist, bins = np.histogram(df["steering"], n_bins)
    center = (bins[:-1] + bins[1:]) * 0.5

    if display:
        plt.figure()
        plt.bar(center, hist, width=0.05)
        plt.title("Steering Distribution BEFORE balancing")
        plt.xlabel("Steering")
        plt.ylabel("Count")
        plt.show()

    remove_index_list = []

    for j in range(n_bins):
        bin_indices = df[
            (df["steering"] >= bins[j]) & (df["steering"] <= bins[j + 1])
        ].index.values

        bin_indices = shuffle(bin_indices, random_state=42)

        if len(bin_indices) > samples_per_bin:
            remove_index_list.extend(bin_indices[samples_per_bin:])

    print("Removed images:", len(remove_index_list))
    df_balanced = df.drop(index=remove_index_list).reset_index(drop=True)
    print("Remaining images:", len(df_balanced))

    if display:
        hist, _ = np.histogram(df_balanced["steering"], n_bins)
        plt.figure()
        plt.bar(center, hist, width=0.05)
        plt.plot(
            (np.min(df_balanced["steering"]), np.max(df_balanced["steering"])),
            (samples_per_bin, samples_per_bin),
        )
        plt.title("Steering Distribution AFTER balancing")
        plt.xlabel("Steering")
        plt.ylabel("Count")
        plt.show()

    return df_balanced

def load_image_paths_and_steering(data_dir, csv_path):
    """
    Build arrays of image paths (center camera) and steering values
    from a balanced driving_log CSV.

    data_dir: base folder (e.g. "data")
    csv_path: path to CSV (e.g. "data/merged_driving_log_balanced.csv")
    """
    df = pd.read_csv(csv_path)

    image_paths = []
    steerings = []

    for _, row in df.iterrows():
        # Original center column may contain an absolute path.
        # We only keep the filename and rebuild the path.
        filename = os.path.basename(row["center"].strip())
        run_folder = row["run"]  # e.g. "run1_clean"

        full_path = os.path.join(data_dir, run_folder, "IMG", filename)

        image_paths.append(full_path)
        steerings.append(float(row["steering"]))

    return np.asarray(image_paths), np.asarray(steerings)

# ---------- Image preprocessing ----------
def preprocess_image(img):
    """
    NVIDIA-style preprocessing:
    - Crop sky and hood
    - Convert RGB -> YUV
    - Gaussian blur
    - Resize to 200x66
    - Normalize to [0, 1]
    """
    # crop
    img = img[60:135, :, :]

    # color space, separates brightness from color information
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # resize (width=200, height=66)
    img = cv2.resize(img, (200, 66))

    # normalize
    img = img / 255.0

    return img


# ---------- Data augmentation helpers ----------
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 0.3 + np.random.rand() * 0.7   # 0.3–1.0
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image


def random_translate(image, steering, range_x=50, range_y=10):
    """
    Random shift in x/y. Adjust steering slightly based on x-shift.
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)

    steering += trans_x * 0.002  # small factor, can tweak

    trans_mat = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_mat, (width, height))

    return image, steering


def random_flip(image, steering):
    """
    Horizontal flip with 50% probability.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering
    return image, steering


def augment_image(image, steering):
    """
    Apply a series of random augmentations.
    Called only for training samples.
    """
    # brightness
    if np.random.rand() < 0.5:
        image = random_brightness(image)

    # translation
    if np.random.rand() < 0.5:
        image, steering = random_translate(image, steering, 50, 10)

    # flip
    image, steering = random_flip(image, steering)

    return image, steering


# ---------- Batch generator ----------
def batch_generator(image_paths, steerings, batch_size=64, is_training=True):
    """
    image_paths: np.array of image file paths
    steerings:   np.array of steering angles

    Yields batches (X, y) ready for model.fit().
    """
    num_samples = len(image_paths)

    while True:
        image_paths, steerings = shuffle(image_paths, steerings)

        for offset in range(0, num_samples, batch_size):
            batch_paths = image_paths[offset:offset + batch_size]
            batch_steers = steerings[offset:offset + batch_size]

            images = []
            angles = []

            for path, steering in zip(batch_paths, batch_steers):
                # load image (cv2 loads BGR)
                img = cv2.imread(path)
                if img is None:
                    # bad path, skip
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if is_training:
                    # apply augmentation
                    img, steering_aug = augment_image(img, float(steering))
                else:
                    steering_aug = float(steering)

                # preprocess after augmentation
                img = preprocess_image(img)

                images.append(img)
                angles.append(steering_aug)

            if len(images) == 0:
                continue

            X = np.array(images)
            y = np.array(angles)

            yield X, y