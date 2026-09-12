# scripts/save_class_names.py
import os, json
from pathlib import Path

DATA_DIR = Path("data/raw/PlantVillage")
class_names = sorted(os.listdir(DATA_DIR))

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

print(f"Saved {len(class_names)} classes.")