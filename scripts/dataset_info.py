import os

DATASET_PATH = "../dataset/food20"

total_images = 0

for cls in sorted(os.listdir(DATASET_PATH)):

    class_path = os.path.join(DATASET_PATH, cls)

    count = len(os.listdir(class_path))

    total_images += count

    print(f"{cls}: {count}")

print()
print("Toplam Görsel:", total_images)