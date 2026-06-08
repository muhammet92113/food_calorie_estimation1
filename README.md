# Food Image Classification and Calorie Estimation

## Project Overview

This project compares three different approaches for food image classification:

1. Color Histogram + KNN (Baseline)
2. HOG + SVM (Classical Machine Learning)
3. MobileNetV3 and ResNet50 (Deep Learning)

The dataset is a 20-class subset of Food-101.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate environment:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run:

```bash
python train.py
```

## Testing

Run:

```bash
python test.py
```

## Outputs

Generated results are stored in:

* outputs/training_curve.png
* outputs/confusion_matrix.png
* outputs/error_samples.png

## Dataset

Food-101 subset with 20 classes.
