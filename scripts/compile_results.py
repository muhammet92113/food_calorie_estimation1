"""Build final results table from checkpoints and cached classical ML."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from dataset.dataset import get_dataloaders
from methods.baseline import extract_color_histogram, run_baseline
from methods.data_utils import load_split_paths
from methods.deep_learning import _collect_predictions, _get_device
from models.mobilenet import get_mobilenet
from models.resnet import get_resnet50
from utils.metrics import evaluate_predictions, save_results_table
from utils.paths import MODELS_DIR, RESULTS_DIR

CACHE_DIR = RESULTS_DIR / "hog_cache"


def eval_hog_cached():
    train_paths, train_labels, class_names = load_split_paths("train")
    test_paths, test_labels, _ = load_split_paths("test")
    X_train = joblib.load(CACHE_DIR / "hog_train.joblib")
    X_test = joblib.load(CACHE_DIR / "hog_test.joblib")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=42)
    clf.fit(X_train, train_labels)
    y_pred = clf.predict(X_test)
    metrics = evaluate_predictions(test_labels, y_pred, class_names)
    return {
        "method": "HOG+SVM",
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "train_time_min": 3.5,
    }


def eval_checkpoint(model_fn, ckpt_name, method_name, train_time_min):
    _, _, test_loader, class_names = get_dataloaders(augment=True)
    model = model_fn(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODELS_DIR / ckpt_name, map_location="cpu", weights_only=True))
    device = _get_device()
    model = model.to(device)
    y_true, y_pred = _collect_predictions(model, test_loader, device)
    metrics = evaluate_predictions(y_true, y_pred, class_names)
    return {
        "method": method_name,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "train_time_min": train_time_min,
    }


def main():
    results = []

    b = run_baseline()
    results.append({k: b[k] for k in ("method", "accuracy", "f1_macro", "train_time_min")})

    if (CACHE_DIR / "hog_train.joblib").exists():
        results.append(eval_hog_cached())
    else:
        from methods.hog_svm import run_hog_svm
        h = run_hog_svm()
        results.append({k: h[k] for k in ("method", "accuracy", "f1_macro", "train_time_min")})

    if (MODELS_DIR / "mobilenetv3.pth").exists():
        results.append(eval_checkpoint(get_mobilenet, "mobilenetv3.pth", "MobileNetV3", 32))

    if (MODELS_DIR / "resnet50.pth").exists():
        results.append(eval_checkpoint(get_resnet50, "resnet50.pth", "ResNet50", 20.2))

    save_results_table(results, RESULTS_DIR / "results_table")


if __name__ == "__main__":
    main()
