# Food Image Classification and Calorie Estimation

This project implements a comparative study of different approaches for food image classification and calorie estimation. The system evaluates classical machine learning methods and deep learning models on a 20-class subset of the Food-101 dataset.

---

## Project Overview

The project compares the following approaches:

- Baseline Method: Color Histogram + KNN
- Classical Machine Learning: HOG + SVM
- Deep Learning: MobileNetV3 and ResNet50 (Transfer Learning)

The main objective is to analyze performance differences in terms of accuracy, computational cost, and feature representation capability.

---

## Project Structure

methods/        → Baseline, HOG-SVM, Deep Learning implementations  
models/         → MobileNetV3 and ResNet50 model definitions  
scripts/        → Dataset preparation and utility scripts  
utils/          → Helper functions (metrics, paths, calorie mapping)  
results/        → Evaluation outputs (confusion matrices, logs, reports)  
outputs/        → Training curves and visualizations  

---

## Dataset (IMPORTANT)

This project uses a subset of the Food-101 dataset (20 classes).

❗ Dataset is NOT included in this repository due to size limitations.

### How to Download Dataset

Run:

python scripts/download_dataset.py

Then split dataset:

python scripts/split_dataset.py

Final structure:

dataset/
└── food20_split/
    ├── train/
    ├── val/
    └── test/

---

## Installation

Create virtual environment:

python -m venv .venv

Activate environment (Windows):

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## Training

python train.py

---

## Testing / Evaluation

python test.py

---

## Outputs

After training, the following outputs are generated:

- outputs/training_curve.png → Loss and accuracy curves  
- results/confusion_*.png → Confusion matrices for each model  
- results/class_distribution.png → Dataset distribution  
- results/error_samples.png → Misclassified samples  

---

## Experiment Logs

Automatically generated logs:

- results/run_log.txt → Full training logs  
- results/run_hog.txt → HOG-SVM logs  
- results/results_table.txt → Final evaluation table  

Hardware (GPU/CPU) and hyperparameters are also recorded.

---

## Important Notes

- Large files (dataset, checkpoints, cache) are excluded using .gitignore  
- Project is fully reproducible from scripts  
- Only code + results are stored in GitHub  

---

## Citation

If you use this project, please cite relevant related works in your report.

---

## Author

Computer Engineering Project – 2026