import json
import shutil
from collections import Counter
from pathlib import Path

from utils.paths import RESULTS_DIR


def run_error_analysis(result, max_examples=15):
    """Save misclassified examples and confusion pair statistics."""
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    class_names = result["class_names"]
    test_paths = result.get("test_paths")

    errors = []
    pair_counts = Counter()

    for i, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred)):
        if true_idx != pred_idx:
            true_name = class_names[true_idx]
            pred_name = class_names[pred_idx]
            pair = f"{true_name} -> {pred_name}"
            pair_counts[pair] += 1
            if test_paths:
                errors.append({
                    "true": true_name,
                    "predicted": pred_name,
                    "path": test_paths[i],
                })

    error_dir = RESULTS_DIR / "error_analysis"
    error_dir.mkdir(exist_ok=True)

    top_pairs = pair_counts.most_common(20)
    summary = {
        "total_errors": len(errors),
        "top_confusion_pairs": [
            {"pair": pair, "count": count} for pair, count in top_pairs
        ],
        "sample_errors": errors[:max_examples],
    }

    with open(error_dir / "error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    examples_dir = error_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    for ex in errors[:max_examples]:
        src = Path(ex["path"])
        dst_name = f"{ex['true']}_as_{ex['predicted']}_{src.name}"
        shutil.copy(src, examples_dir / dst_name)

    print(f"\nError analysis: {len(errors)} misclassifications")
    print("Top confusion pairs:")
    for pair, count in top_pairs[:5]:
        print(f"  {pair}: {count}")

    return summary
