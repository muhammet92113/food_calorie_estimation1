import time

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from methods.data_utils import load_split_paths
from utils.metrics import evaluate_predictions, save_confusion_matrix
from utils.paths import RESULTS_DIR


def extract_color_histogram(image_path, bins=32):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(bins * 3)

    img = cv2.resize(img, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    return hist / (hist.sum() + 1e-7)


def run_baseline(k=5):
    print("\n=== Baseline: Color Histogram + KNN ===")
    start = time.time()

    train_paths, train_labels, class_names = load_split_paths("train")
    test_paths, test_labels, _ = load_split_paths("test")

    print(f"Extracting histogram features ({len(train_paths)} train, {len(test_paths)} test)...")
    X_train = np.array([extract_color_histogram(p) for p in tqdm(train_paths, desc="Hist train")])
    X_test = np.array([extract_color_histogram(p) for p in tqdm(test_paths, desc="Hist test")])

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, train_labels)
    y_pred = clf.predict(X_test)

    elapsed = (time.time() - start) / 60
    metrics = evaluate_predictions(test_labels, y_pred, class_names)
    print(metrics["report"])
    print(f"Train time: {elapsed:.1f} min")

    save_confusion_matrix(
        test_labels, y_pred, class_names,
        RESULTS_DIR / "confusion_baseline.png",
    )

    return {
        "method": "Baseline Histogram+KNN",
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "train_time_min": round(elapsed, 2),
        "y_true": test_labels,
        "y_pred": y_pred.tolist(),
        "class_names": class_names,
        "test_paths": test_paths,
    }
