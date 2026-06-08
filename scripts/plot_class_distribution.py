"""Plot class distribution for the report Dataset section."""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.paths import SPLIT_DIR

splits = ["train", "val", "test"]
data = {s: {} for s in splits}

for split in splits:
    split_dir = SPLIT_DIR / split
    for cls_dir in sorted(split_dir.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.iterdir()))
            data[split][cls_dir.name] = count

classes = sorted(data["train"].keys())
x = range(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar([i - width for i in x], [data["train"][c] for c in classes], width, label="Train")
ax.bar(x, [data["val"][c] for c in classes], width, label="Val")
ax.bar([i + width for i in x], [data["test"][c] for c in classes], width, label="Test")

ax.set_xlabel("Food Class")
ax.set_ylabel("Image Count")
ax.set_title("Class Distribution (Food-20 Subset)")
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(SPLIT_DIR.parent.parent / "results" / "class_distribution.png", dpi=150)
print("Saved: results/class_distribution.png")
