    """
    Food Type Recognition — Full Experiment Pipeline
    
    Runs all methods required for the final project:
      1. Baseline (Color Histogram + KNN)
      2. Classical ML (HOG + SVM)
      3. MobileNetV3-Large (transfer learning)
      4. ResNet50 (transfer learning)
      5. Ablation studies (augmentation, fine-tuning)
      6. Error analysis on best model
    """
    import argparse
    import json

    from methods.baseline import run_baseline
    from methods.deep_learning import train_deep_model
    from methods.error_analysis import run_error_analysis
    from methods.hog_svm import run_hog_svm
    from models.mobilenet import get_mobilenet
    from models.resnet import get_resnet50
    from utils.metrics import save_results_table
    from utils.paths import RESULTS_DIR


    def run_main_methods(epochs=5):
        results = []

        baseline_result = run_baseline()
        results.append({k: v for k, v in baseline_result.items() if k not in ("y_true", "y_pred", "class_names", "test_paths")})

        hog_result = run_hog_svm()
        results.append({k: v for k, v in hog_result.items() if k not in ("y_true", "y_pred", "class_names", "test_paths")})

        mobilenet_result = train_deep_model(
            get_mobilenet(num_classes=20, freeze_features=True),
            method_name="MobileNetV3",
            epochs=epochs,
            augment=True,
            save_name="mobilenetv3",
        )
        results.append({k: v for k, v in mobilenet_result.items() if k not in ("y_true", "y_pred", "class_names")})

        resnet_result = train_deep_model(
            get_resnet50(num_classes=20, freeze_features=True),
            method_name="ResNet50",
            epochs=epochs,
            augment=True,
            save_name="resnet50",
        )
        results.append({k: v for k, v in resnet_result.items() if k not in ("y_true", "y_pred", "class_names")})

        save_results_table(results, RESULTS_DIR / "results_table")
        run_error_analysis(resnet_result)

        return results


    def run_ablation(epochs=5):
        """Ablation: augmentation on/off and head-only vs full fine-tuning."""
        ablation = []

        no_aug = train_deep_model(
            get_resnet50(num_classes=20, freeze_features=True),
            method_name="ResNet50 (No Aug)",
            epochs=epochs,
            augment=False,
            save_name="resnet50_no_aug",
        )
        ablation.append({
            "experiment": "augmentation",
            "setting": "no_augmentation",
            "accuracy": no_aug["accuracy"],
            "f1_macro": no_aug["f1_macro"],
        })

        with_aug = train_deep_model(
            get_resnet50(num_classes=20, freeze_features=True),
            method_name="ResNet50 (With Aug)",
            epochs=epochs,
            augment=True,
            save_name="resnet50_with_aug",
        )
        ablation.append({
            "experiment": "augmentation",
            "setting": "with_augmentation",
            "accuracy": with_aug["accuracy"],
            "f1_macro": with_aug["f1_macro"],
        })

        head_only = train_deep_model(
            get_resnet50(num_classes=20, freeze_features=True),
            method_name="ResNet50 (Head Only)",
            epochs=epochs,
            augment=True,
            save_name="resnet50_head_only",
        )
        ablation.append({
            "experiment": "fine_tuning",
            "setting": "head_only",
            "accuracy": head_only["accuracy"],
            "f1_macro": head_only["f1_macro"],
        })

        full_ft = train_deep_model(
            get_resnet50(num_classes=20, freeze_features=False),
            method_name="ResNet50 (Full FT)",
            epochs=epochs,
            augment=True,
            save_name="resnet50_full_ft",
        )
        ablation.append({
            "experiment": "fine_tuning",
            "setting": "full_network",
            "accuracy": full_ft["accuracy"],
            "f1_macro": full_ft["f1_macro"],
        })

        with open(RESULTS_DIR / "ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(ablation, f, indent=2)

        print("\n=== Ablation Results ===")
        for row in ablation:
            print(f"  {row['experiment']}/{row['setting']}: {row['accuracy']:.1f}%")

        return ablation


    def main():
        parser = argparse.ArgumentParser(description="Food recognition experiment pipeline")
        parser.add_argument("--epochs", type=int, default=5, help="Training epochs for DL models")
        parser.add_argument("--only", choices=["baseline", "hog", "mobilenet", "resnet"], help="Run single method")
        parser.add_argument("--ablation", action="store_true", help="Run ablation studies")
        args = parser.parse_args()

        if args.only == "baseline":
            run_baseline()
        elif args.only == "hog":
            run_hog_svm()
        elif args.only == "mobilenet":
            train_deep_model(
                get_mobilenet(num_classes=20), "MobileNetV3",
                epochs=args.epochs, augment=True, save_name="mobilenetv3",
            )
        elif args.only == "resnet":
            train_deep_model(
                get_resnet50(num_classes=20), "ResNet50",
                epochs=args.epochs, augment=True, save_name="resnet50",
            )
        elif args.ablation:
            run_ablation(epochs=args.epochs)
        else:
            run_main_methods(epochs=args.epochs)

        print(f"\nResults saved to: {RESULTS_DIR}")


    if __name__ == "__main__":
        main()
