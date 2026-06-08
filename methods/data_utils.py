import os
from pathlib import Path

from utils.paths import SPLIT_DIR


def load_split_paths(split="train"):
    """Return (file_paths, labels, class_names) from food20_split."""
    split_dir = SPLIT_DIR / split
    class_names = sorted(
        d.name for d in split_dir.iterdir() if d.is_dir()
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    paths, labels = [], []
    for cls in class_names:
        cls_dir = split_dir / cls
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(img_path))
                labels.append(class_to_idx[cls])

    return paths, labels, class_names
