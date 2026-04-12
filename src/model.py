# model.py
# Defines the CNN architecture for brain tumor classification.
# Input: 224x224 RGB MRI image
# Output: probability of tumor presence (0 = no tumor, 1 = tumor)

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3)):
    """
    Builds a convolutional neural network for binary image classification.
    
    Architecture:
        3 convolutional blocks (Conv2D + ReLU + MaxPooling)
        Flatten + Dense layers
        Dropout for regularization
        Sigmoid output for binary classification
    """
    model = models.Sequential([

        # Block 1 — learn basic edges and textures
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(2, 2),

        # Block 2 — learn more complex patterns
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        # Block 3 — learn high-level features
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        # Flatten and classify
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # Prevents overfitting on small medical datasets
        layers.Dense(1, activation="sigmoid")  # Binary output
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall()]
        # Recall is tracked because in medical imaging,
        # missing a tumor (false negative) is worse than a false alarm
    )

    return model


def summary():
    """Prints the model architecture to the console."""
    model = build_model()
    model.summary()


if __name__ == "__main__":
    summary()
