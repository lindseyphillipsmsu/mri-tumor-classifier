# evaluate.py
# Evaluates trained model performance and generates visual results.
# Outputs: confusion matrix, ROC curve, classification report.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)


def evaluate_model(model, X_test, y_test):
    """
    Runs full evaluation on test data.
    Prints classification report and plots confusion matrix + ROC curve.
    """
    print("Evaluating model...\n")

    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

    # Classification report
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["No Tumor", "Tumor"]
    ))

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # ROC curve
    plot_roc_curve(y_test, y_pred_prob)


def plot_confusion_matrix(y_test, y_pred):
    """
    Plots and saves confusion matrix.
    Shows true positives, false positives, false negatives, true negatives.
    In medical imaging, false negatives are the critical failure case.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Tumor", "Tumor"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — MRI Tumor Classifier")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to results/")


def plot_roc_curve(y_test, y_pred_prob):
    """
    Plots and saves ROC curve.
    AUC closer to 1.0 = better model.
    Random guessing = 0.5 (the diagonal baseline).
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#cba6f7", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#6c6888", lw=1,
             linestyle="--", label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve — MRI Tumor Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=150)
    plt.show()
    print(f"ROC curve saved to results/ | AUC: {roc_auc:.3f}")


def print_metrics_summary(model, X_test, y_test):
    """
    Prints a clean summary of key metrics.
    """
    loss, accuracy, recall = model.evaluate(X_test, y_test, verbose=0)
    print("\n── Model Performance ──────────────────")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Recall   : {recall:.4f}  ← most important in medical imaging")
    print(f"  Loss     : {loss:.4f}")
    print("────────────────────────────────────────")
