"""Quick MobileNet training script (use run_all.py for full pipeline)."""
from methods.deep_learning import train_deep_model
from models.mobilenet import get_mobilenet

if __name__ == "__main__":
    train_deep_model(
        get_mobilenet(num_classes=20, freeze_features=True),
        method_name="MobileNetV3",
        epochs=5,
        augment=True,
        save_name="mobilenetv3",
    )
