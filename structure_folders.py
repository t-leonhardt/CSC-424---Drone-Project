import os
import random
import shutil
from pathlib import Path

# Paths
source_dir = Path(r"C:\Users\tomin\Documents\images")  # Folder with all 3,000 images
dest_dir = Path("solar_panel_dataset")    # New organized dataset

# Create subfolders
(dest_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
(dest_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
(dest_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
(dest_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

# Split images into train/val (80% train, 20% val)
all_images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))  
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))  # 80% for training

for i, img in enumerate(all_images):
    if i < split_idx:
        shutil.copy(img, dest_dir / "images" / "train" / img.name)
    else:
        shutil.copy(img, dest_dir / "images" / "val" / img.name)