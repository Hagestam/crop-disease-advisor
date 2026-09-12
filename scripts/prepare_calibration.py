"""
Copies 100 random images from the dataset into data/calibration/
for use as the INT8 representative dataset.
"""

import os, shutil, random
from pathlib import Path

SRC = Path("data/raw/PlantVillage")
DST = Path("data/calibration")
N   = 200  # number of calibration images

DST.mkdir(exist_ok=True)

all_images = []
for cls_dir in SRC.iterdir():
    if cls_dir.is_dir():
        all_images.extend(list(cls_dir.glob("*.jpg")))
        all_images.extend(list(cls_dir.glob("*.JPG")))
        all_images.extend(list(cls_dir.glob("*.png")))

random.shuffle(all_images)
selected = all_images[:N]

for i, img_path in enumerate(selected):
    shutil.copy(img_path, DST / f"calib_{i:04d}.jpg")

print(f"Copied {len(selected)} calibration images to {DST}")