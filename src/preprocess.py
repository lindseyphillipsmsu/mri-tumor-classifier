# preprocess.py
# Loads MRI images, resizes them, normalizes pixel values,
# and returns arrays ready for model training.

import os
import numpy as np
import cv2

IMG_SIZE = 224  # Standard input size for CNNs

def load_images(data_dir):
    """
    Loads images from /data/yes and /data/no folders.
    Returns numpy arrays of images and their labels.
    """
    images = []
    labels = []

    categories = {"yes": 1, "no": 0}  # 1 = tumor, 0 = no tumor

    for category, label in categories.items():
        folder = os.path.join(data_dir, category)

        if not os.path.exists(folder):
            print(f"Warning: folder not found — {folder}")
            continue

        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Could not read: {img_path}")
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)


def normalize(images):
    """
    Normalizes pixel values from 0-255 to 0-1.
    Neural networks train faster and more stably on normalized data.
    """
    return images.astype("float32") / 255.0


def preprocess(data_dir):
    """
    Full pipeline: load → normalize → return ready-to-train data.
    """
    print("Loading images...")
    images, labels = load_images(data_dir)

    print(f"Loaded {len(images)} images")
    print(f"Tumors: {sum(labels)} | No tumor: {len(labels) - sum(labels)}")

    images = normalize(images)
    print("Normalization complete.")

    return images, labels
