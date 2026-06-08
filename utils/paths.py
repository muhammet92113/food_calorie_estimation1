from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
SPLIT_DIR = DATA_DIR / "food20_split"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "checkpoints"
CALORIE_JSON = PROJECT_ROOT / "calorie_data" / "calories.json"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
