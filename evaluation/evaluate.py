import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

"""
Model evaluation utilities.

This module provides the following evaluation methods:

1. Confusion Matrix
   Visualizes how often classes are correctly classified
   or confused with one another.

2. Accuracy Comparison
   Compares the accuracy of all models using a bar chart.

3. Runtime Comparison
   Compares training and inference times across models.

4. Classification Report
   Displays Precision, Recall, and F1-Score for each class.
"""


def plot_confusion_matrix(y_test, predictions, model_name, class_names):
    """
    Plot and save the confusion matrix for a model.

    Parameters
    ----------
    y_test : np.ndarray
        Ground-truth labels.
    predictions : np.ndarray
        Predicted labels.
    model_name : str
        Name of the model.
    class_names : list
        List of class names.
    """
    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    # Display confusion matrix using a blue color scale
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title(
        f"Confusion Matrix — {model_name}",
        fontsize=13,
        fontweight="bold"
    )

    # Automatically adjust spacing between plot elements
    plt.tight_layout()

    # Save plot as PNG
    plt.savefig(
        f"confusion_matrix_{model_name}.png",
        dpi=150
    )

    # Close figure after saving
    plt.close()

    print(
        f"    Confusion Matrix saved: "
        f"confusion_matrix_{model_name}.png"
    )


# =========================
# Model Comparison Plot
# =========================

def plot_model_comparison(results):
    """
    Plot accuracy, training time, and inference time
    for all models side by side.

    Parameters
    ----------
    results : dict
        Dictionary containing model results.
    """
    models = list(results.keys())

    accuracies = [
        results[m]["accuracy"] * 100
        for m in models
    ]

    train_times = [
        results[m]["training_time"]
        for m in models
    ]

    inf_times = [
        results[m]["inference_time"]
        for m in models
    ]

    # Create three plots side by side
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    colors = [
        "steelblue",
        "seagreen",
        "tomato"
    ]

    # =========================
    # Plot 1 — Accuracy
    # =========================
    axes[0].bar(
        models,
        accuracies,
        color=colors
    )

    axes[0].set_title(
        "Accuracy (%)",
        fontweight="bold"
    )

    axes[0].set_ylabel(
        "Accuracy (%)"
    )

    axes[0].set_ylim(0, 110)

    for i, value in enumerate(accuracies):
        axes[0].text(
            i,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontweight="bold"
        )

    axes[0].grid(
        axis="y",
        alpha=0.3
    )

    # =========================
    # Plot 2 — Training Time
    # =========================
    axes[1].bar(
        models,
        train_times,
        color=colors
    )

    axes[1].set_title(
        "Training Time (s)",
        fontweight="bold"
    )

    axes[1].set_ylabel(
        "Seconds"
    )

    for i, value in enumerate(train_times):
        axes[1].text(
            i,
            value + 0.01,
            f"{value:.4f}s",
            ha="center",
            fontweight="bold",
            fontsize=9
        )

    axes[1].grid(
        axis="y",
        alpha=0.3
    )

    # =========================
    # Plot 3 — Inference Time
    # =========================
    axes[2].bar(
        models,
        inf_times,
        color=colors
    )

    axes[2].set_title(
        "Inference Time (s)",
        fontweight="bold"
    )

    axes[2].set_ylabel(
        "Seconds"
    )

    for i, value in enumerate(inf_times):
        axes[2].text(
            i,
            value + 0.0001,
            f"{value:.4f}s",
            ha="center",
            fontweight="bold",
            fontsize=9
        )

    axes[2].grid(
        axis="y",
        alpha=0.3
    )

    # Overall title for all plots
    plt.suptitle(
        "Model Comparison: " + " vs. ".join(models),
        fontsize=14,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    plt.savefig(
        "model_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "    Comparison plot saved: "
        "model_comparison.png"
    )


# =========================
# Classification Report
# =========================

def print_classification_report(
    y_test,
    predictions,
    model_name,
    class_names
):
    """
    Print Precision, Recall, and F1-Score
    for each class.

    Precision
        Of all samples predicted as a class,
        how many were correct?

    Recall
        Of all true samples belonging to a class,
        how many were correctly identified?

    F1-Score
        Harmonic mean of Precision and Recall.
    """
    print(
        f"\n  Classification Report — "
        f"{model_name}"
    )

    print("-" * 50)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=class_names
        )
    )


# =========================
# Main Evaluation Function
# =========================

def evaluate_all(
    results_with_predictions,
    y_test,
    class_names
):
    """
    Run all evaluation methods for all models.

    Parameters
    ----------
    results_with_predictions : dict
        Dictionary containing model results
        and predictions.

    y_test : np.ndarray
        Ground-truth labels.

    class_names : list
        List of class names.
    """
    print("\n[Evaluation] Starting evaluation...")

    # Generate confusion matrix and classification report
    # for each model
    print("\n  Confusion Matrices:")

    for model_name, data in results_with_predictions.items():

        plot_confusion_matrix(
            y_test,
            data["predictions"],
            model_name,
            class_names
        )

        print_classification_report(
            y_test,
            data["predictions"],
            model_name,
            class_names
        )

    # Generate model comparison plot
    print("\n  Model Comparison:")

    plot_model_comparison(
        results_with_predictions
    )

    print("\n[Evaluation] Completed.")