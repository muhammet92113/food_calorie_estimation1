import json

from utils.paths import CALORIE_JSON


def load_calorie_table():
    with open(CALORIE_JSON, encoding="utf-8") as f:
        return json.load(f)


def estimate_calories(food_class, weight_g):
    """Estimate total kcal: (kcal_per_100g / 100) * weight_g"""
    table = load_calorie_table()
    if food_class not in table:
        raise ValueError(f"Unknown food class: {food_class}")
    kcal_per_100g = table[food_class]
    return round((kcal_per_100g / 100) * weight_g, 1)
