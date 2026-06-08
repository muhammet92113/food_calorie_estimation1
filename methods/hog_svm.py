import time

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from methods.data_utils import load_split_paths
from utils.metrics import evaluate_predictions, save_confusion_matrix
from utils.paths import RESULTS_DIR

CACHE_DIR = RESULTS_DIR / "hog_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extract_hog(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(1764)

    img = cv2.resize(img, (128, 128))
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


def _load_or_extract(paths, cache_name):
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        print(f"Loading cached features: {cache_name}")
        return joblib.load(cache_path)
    features = np.array([extract_hog(p) for p in tqdm(paths, desc=cache_name)])
    joblib.dump(features, cache_path)
    return features


def run_hog_svm():
    print("\n=== Classical ML: HOG + SVM ===")
    start = time.time()

    train_paths, train_labels, class_names = load_split_paths("train")
    test_paths, test_labels, _ = load_split_paths("test")

    X_train = _load_or_extract(train_paths, "hog_train.joblib")
    X_test = _load_or_extract(test_paths, "hog_test.joblib")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training linear SVM (SGD)...")
    clf = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=42)
    clf.fit(X_train, train_labels)
    y_pred = clf.predict(X_test)

    elapsed = (time.time() - start) / 60
    metrics = evaluate_predictions(test_labels, y_pred, class_names)
    print(metrics["report"])
    print(f"Train time: {elapsed:.1f} min")

    save_confusion_matrix(
        test_labels, y_pred, class_names,
        RESULTS_DIR / "confusion_hog_svm.png",
    )

    return {
        "method": "HOG+SVM",
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "train_time_min": round(elapsed, 2),
        "y_true": test_labels,
        "y_pred": y_pred.tolist(),
        "class_names": class_names,
        "test_paths": test_paths,
    }
