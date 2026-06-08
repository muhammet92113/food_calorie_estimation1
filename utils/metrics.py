import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_predictions(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    return {
        "accuracy": round(accuracy * 100, 2),
        "f1_macro": round(f1, 4),
        "report": report,
    }


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_results_table(results, save_path):
    """results: list of dicts with keys method, accuracy, f1_macro, train_time_min"""
    save_path = Path(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    header = f"{'Method':<30} {'Accuracy':>10} {'F1':>8} {'Train Time':>12}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r['method']:<30} {r['accuracy']:>9.1f}% {r['f1_macro']:>8.2f} "
            f"{r['train_time_min']:>10.1f} min"
        )
    text = "\n".join(lines)
    print(text)
    with open(save_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text
