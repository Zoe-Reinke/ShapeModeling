import shutil
from pathlib import Path

path = Path("/Users/zoereinke/Repositories/ShapeModeling/Thyroid/Totalsegmentator_dataset_v201")

for s in path.iterdir():  # s = case
    if s.is_dir():
        if not (s / "thyroid_left.nii.gz").exists() or not (s / "thyroid_right.nii.gz").exists():
            continue
        segmentations = s / "segmentations"
        left = s / "thyroid_left.nii.gz"
        right = s / "thyroid_right.nii.gz"

        #shutil.move(left, segmentations / "thyroid_left.nii.gz")
        #shutil.move(right, segmentations / "thyroid_right.nii.gz")