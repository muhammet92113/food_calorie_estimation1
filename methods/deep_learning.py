import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset import get_dataloaders
from utils.metrics import evaluate_predictions, save_confusion_matrix
from utils.paths import MODELS_DIR, RESULTS_DIR


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def train_deep_model(
    model,
    method_name,
    epochs=5,
    lr=0.001,
    augment=True,
    save_name=None,
):
    print(f"\n=== Deep Learning: {method_name} (aug={augment}) ===")
    device = _get_device()
    print(f"Device: {device}")

    start = time.time()
    train_loader, val_loader, test_loader, class_names = get_dataloaders(augment=augment)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {total_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%"
        )

    if best_state:
        model.load_state_dict(best_state)

    elapsed = (time.time() - start) / 60

    y_true, y_pred = _collect_predictions(model, test_loader, device)
    metrics = evaluate_predictions(y_true, y_pred, class_names)
    print(metrics["report"])
    print(f"Test Accuracy: {metrics['accuracy']:.1f}% | Train time: {elapsed:.1f} min")

    safe_name = (save_name or method_name).replace(" ", "_").replace("+", "").lower()
    torch.save(model.state_dict(), MODELS_DIR / f"{safe_name}.pth")
    save_confusion_matrix(
        y_true, y_pred, class_names,
        RESULTS_DIR / f"confusion_{safe_name}.png",
    )

    return {
        "method": method_name,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "train_time_min": round(elapsed, 2),
        "y_true": y_true,
        "y_pred": y_pred,
        "class_names": class_names,
        "val_accuracy": round(best_val_acc, 2),
    }
