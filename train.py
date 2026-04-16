# train.py
# Runs the full training pipeline for the MRI tumor classifier.
# Loads data, splits it, trains the model, saves weights, runs evaluation.

import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess
from src.model import build_model
from src.evaluate import evaluate_model, print_metrics_summary

DATA_DIR = "data"
MODEL_SAVE_PATH = "results/model.keras"
EPOCHS = 20
BATCH_SIZE = 32


def main():
    # load & split
    images, labels = preprocess(DATA_DIR)

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # build & train
    model = build_model()
    model.summary()

    print("\nStarting training...\n")

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight={0: 1.0, 1: 2.0}
    )

    # save
    os.makedirs("results", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # evaluate
    print("\nRunning evaluation on test set...")
    evaluate_model(model, X_test, y_test)
    print_metrics_summary(model, X_test, y_test)


if __name__ == "__main__":
    main()
