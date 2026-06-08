import os
import shutil
from tqdm import tqdm

SOURCE_DIR = "../dataset/food-101/images"
TARGET_DIR = "../dataset/food20"

SELECTED_CLASSES = [
    "pizza",
    "hamburger",
    "sushi",
    "ice_cream",
    "donuts",
    "steak",
    "hot_dog",
    "waffles",
    "pancakes",
    "omelette",
    "greek_salad",
    "french_fries",
    "cheesecake",
    "apple_pie",
    "spaghetti_bolognese",
    "ramen",
    "breakfast_burrito",
    "club_sandwich",
    "fried_rice",
    "chicken_wings"
]

os.makedirs(TARGET_DIR, exist_ok=True)

for food_class in SELECTED_CLASSES:

    source_class = os.path.join(SOURCE_DIR, food_class)
    target_class = os.path.join(TARGET_DIR, food_class)

    os.makedirs(target_class, exist_ok=True)

    files = os.listdir(source_class)

    for file in tqdm(files, desc=food_class):
        shutil.copy(
            os.path.join(source_class, file),
            os.path.join(target_class, file)
        )

print("20 sınıf oluşturuldu.")