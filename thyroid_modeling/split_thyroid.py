import shutil
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

path = Path("/Users/zoereinke/Repositories/ShapeModeling/Thyroid/Totalsegmentator_dataset_v201")

for s in path.iterdir():  # s = case
    segmentations = s / "segmentations"
    print("Case: ", s)
    if segmentations.is_dir():
        thyroid = segmentations / "thyroid_gland.nii.gz"
        trachea = segmentations / "trachea.nii.gz"

        thyroid_data = nib.load(str(thyroid))
        trachea_data = nib.load(str(trachea))
        trachea_img = trachea_data.get_fdata()
        affine = trachea_data.affine

        centroids = []
        z_indices = []
        for z in range(trachea_img.shape[2]):
            sl = trachea_img[:, :, z]
            if sl.sum() > 0:
                ax0_idx, ax1_idx = np.nonzero(sl)  # (axis0, axis1) indices, no relabeling
                centroids.append((ax0_idx.mean(), ax1_idx.mean()))
                z_indices.append(z)

        centroids = np.array(centroids)  # shape (N, 2) -> columns are (axis0, axis1)
        z_vals = np.array(z_indices)

        ax0_fit = np.polyfit(z_vals, centroids[:, 0], deg=1)
        ax1_fit = np.polyfit(z_vals, centroids[:, 1], deg=1)

        def centerline_ax0(z):
            return np.polyval(ax0_fit, z)

        thyroid_mask = thyroid_data.get_fdata() > 0.5
        left_lobe = np.zeros_like(thyroid_mask, dtype=bool)
        right_lobe = np.zeros_like(thyroid_mask, dtype=bool)

        for z in range(thyroid_mask.shape[2]):
            sl = thyroid_mask[:, :, z]
            if sl.sum() == 0:
                continue
            c_ax0 = centerline_ax0(z)
            ax0_idx, ax1_idx = np.nonzero(sl)  # consistent (axis0, axis1) order
            # Side test compares axis-0 index to the centerline's axis-0 position
            is_left = ax0_idx < c_ax0  # flip sign depending on your RAS/LPS convention

            for a0, a1, side in zip(ax0_idx, ax1_idx, is_left):
                if side:
                    left_lobe[a0, a1, z] = True
                else:
                    right_lobe[a0, a1, z] = True

        nib.save(nib.Nifti1Image(left_lobe.astype(np.uint8), affine), segmentations / "thyroid_left.nii.gz")
        nib.save(nib.Nifti1Image(right_lobe.astype(np.uint8), affine), segmentations / "thyroid_right.nii.gz")
        