import os
import shutil
import random
from tqdm import tqdm

SOURCE_DIR = "../dataset/food20"
TARGET_DIR = "../dataset/food20_split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

classes = sorted(os.listdir(SOURCE_DIR))

for cls in classes:
    class_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    train_split = int(len(images) * TRAIN_RATIO)
    val_split = int(len(images) * VAL_RATIO)

    train_files = images[:train_split]
    val_files = images[train_split:train_split + val_split]
    test_files = images[train_split + val_split:]

    for split_name, split_files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files)
    ]:
        split_dir = os.path.join(TARGET_DIR, split_name, cls)
        os.makedirs(split_dir, exist_ok=True)

        for file in tqdm(split_files, desc=f"{cls}-{split_name}"):
            src = os.path.join(class_path, file)
            dst = os.path.join(split_dir, file)
            shutil.copy(src, dst)

print("Dataset split tamamlandı.")