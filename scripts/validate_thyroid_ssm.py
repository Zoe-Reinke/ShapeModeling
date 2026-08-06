#!/usr/bin/env python
"""Held-out validation of the pooled thyroid lobe SSM, with a mirror-aware (side-agnostic) fit.

Deployment-honest counterpart to ``build_thyroid_ssm.py`` (copy-and-edited from
``validate_ssm.py``). It takes lobe masks the model never saw and asks how well the exported
**side-agnostic** model reconstructs each one from its raw mask alone -- exactly the 3D-US
reconstruction use case, where you have a lobe surface but no side label.

The model lives in a single (left) chiral frame, and a similarity fit cannot reflect. So for each
held-out lobe we fit **both** chiralities -- the surface as-is and its mirror -- and keep the lower
surface distance. This makes the fit side-agnostic: the winning chirality *infers* the side (a left
lobe fits best as-is, a right lobe fits best mirrored), so you never supply it. We also report
whether that inferred side matches the file's actual side, as a check that the mirror-aware fit and
the pooled model behave as intended.

  1. load the exported model (``model.npz`` from ``build_thyroid_ssm.py``);
  2. load every ``<case>/segmentations/thyroid_{left,right}.nii.gz`` under each ``--data`` dir;
  3. clean each mask, extract its surface in physical mm;
  4. fit the model to the surface AND to its reflection (``ExportedModel.fit_surface``), keep best;
  5. record the symmetric surface distance (mm) per lobe + the inferred side, and summarise.

Needs **no ShapeWorks** -- ``fit_surface`` builds its own pseudo-correspondence.

Usage (from the repo root)::

    python scripts/validate_thyroid_ssm.py --model thyroid_ssm_output/model.npz \
        --data Thyroid_heldout --out thyroid_validation_output --n-modes 15

See ``python scripts/validate_thyroid_ssm.py --help`` for all flags.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from shape_modeling.analysis import (
    CleanConfig,
    clean_mask,
    extract_surface,
    load_model,
)
from shape_modeling.io.reader import LoadedVolume, VolumeType


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=Path, required=True, help="exported model .npz (export_model)")
    p.add_argument(
        "--data",
        type=Path,
        action="append",
        required=True,
        help="held-out TotalSegmentator-style dir (repeat for several); "
        "expects <case>/segmentations/thyroid_{left,right}.nii.gz",
    )
    p.add_argument("--out", type=Path, default=Path("thyroid_validation_output"), help="out folder")
    p.add_argument(
        "--holdout-list",
        type=Path,
        default=None,
        help="only validate subjects whose case id is in this file (one per line); "
        "use <build-out>/heldout.txt so validation runs on exactly the reserved held-out split",
    )
    p.add_argument(
        "--n-modes", type=int, default=None, help="modes the fit may move (default: all)"
    )
    p.add_argument(
        "--smooth-sigma", type=float, default=0.0, help="clean_mask anti-alias sigma (voxels)"
    )
    p.add_argument("--max-sd", type=float, default=3.0, help="clamp fitted weights to +/- this")
    p.add_argument("--n-restarts", type=int, default=4, help="ICP pose restarts per fit")
    p.add_argument(
        "--volume-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[0.3, 40.0],
        help="QC gate: drop lobes whose volume (cm3) is outside [MIN, MAX] (default thyroid-lobe)",
    )
    p.add_argument(
        "--max-lobes", type=int, default=None, help="cap number of lobes fitted (default all)"
    )
    return p.parse_args()


def load_mask(path: Path) -> tuple[np.ndarray, LoadedVolume]:
    """Read a lobe mask as (array, LoadedVolume) in its native physical frame (no crop/mirror)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    vol = LoadedVolume(
        path=path,
        volume_type=VolumeType.ORGAN_SEG,
        size=img.GetSize(),
        spacing=img.GetSpacing(),
        origin=img.GetOrigin(),
        direction=img.GetDirection(),
        dtype=str(arr.dtype),
        sitk_image=img,
        array=arr,
    )
    return arr, vol


def volume_cm3(arr: np.ndarray, vol: LoadedVolume) -> float:
    voxel_mm3 = float(np.prod(vol.spacing))
    return float((arr > 0).sum()) * voxel_mm3 / 1000.0


def iter_lobes(data_dirs: list[Path], only: set[str] | None = None):
    """Yield (case_id, side, path) for every thyroid_{left,right} mask under the data dirs.

    If ``only`` is given, restrict to subjects whose case-directory name is in it (the held-out
    split from ``build_thyroid_ssm.py``'s ``heldout.txt``).
    """
    for d in data_dirs:
        for cdir in sorted(p for p in d.iterdir() if p.is_dir()):
            if only is not None and cdir.name not in only:
                continue
            seg = cdir / "segmentations"
            for side, name in (("L", "thyroid_left.nii.gz"), ("R", "thyroid_right.nii.gz")):
                path = seg / name
                if path.exists():
                    yield f"{d.name}/{cdir.name}", side, path


def summarise(distances: np.ndarray) -> dict:
    if distances.size == 0:
        return {"n": 0}
    return {
        "n": int(distances.size),
        "median_mm": round(float(np.median(distances)), 4),
        "mean_mm": round(float(distances.mean()), 4),
        "p95_mm": round(float(np.percentile(distances, 95)), 4),
        "max_mm": round(float(distances.max()), 4),
    }


def main() -> None:
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)
    print(f"model: {args.model}  ({model.n_modes} modes, {model.n_points} points, "
          f"units={model.metadata.get('units')}, side_agnostic={model.metadata.get('side_agnostic')})")

    clean_cfg = CleanConfig(smooth_sigma=args.smooth_sigma)
    vol_lo, vol_hi = args.volume_range
    rows: list[dict] = []
    excluded: list[dict] = []
    only = None
    if args.holdout_list is not None:
        only = {ln.strip() for ln in args.holdout_list.read_text().splitlines() if ln.strip()}
        print(f"restricting to {len(only)} held-out subjects from {args.holdout_list}")
    lobes = list(iter_lobes(args.data, only))
    if not lobes:
        raise SystemExit("no thyroid_{left,right}.nii.gz masks found under --data dirs"
                         + (" matching --holdout-list" if only else ""))

    for i, (case_id, side, path) in enumerate(lobes, 1):
        if args.max_lobes is not None and len(rows) >= args.max_lobes:
            break
        arr, vol = load_mask(path)
        vcm3 = volume_cm3(arr, vol)
        tag = f"{case_id}_{side}"
        if not (vol_lo <= vcm3 <= vol_hi):
            reason = f"volume {vcm3:.1f} cm3 outside [{vol_lo:g}, {vol_hi:g}]"
            excluded.append({"case_id": tag, "reason": reason})
            print(f"  [{i}/{len(lobes)}] {tag}: {reason} -- skipped (QC)")
            continue
        mask = clean_mask(arr > 0, clean_cfg)
        if not mask.any():
            excluded.append({"case_id": tag, "reason": "empty after cleaning"})
            print(f"  [{i}/{len(lobes)}] {tag}: empty after cleaning -- skipped")
            continue
        target = extract_surface(mask, vol).verts  # (N, 3) physical-mm point cloud

        # Mirror-aware, side-agnostic fit: fit the surface as-is AND reflected; keep the better.
        # The model is left-chiral, so "as-is" wins for a left lobe and "mirrored" for a right one
        # -> the winning chirality infers the side without us supplying it.
        mirrored = target.copy()
        mirrored[:, 0] = -mirrored[:, 0]  # reflect across x -> flip handedness
        fit_asis = model.fit_surface(
            target, n_modes=args.n_modes, max_sd=args.max_sd, n_restarts=args.n_restarts
        )
        fit_mir = model.fit_surface(
            mirrored, n_modes=args.n_modes, max_sd=args.max_sd, n_restarts=args.n_restarts
        )
        used_mirror = fit_mir.mean_distance < fit_asis.mean_distance
        fit = fit_mir if used_mirror else fit_asis
        inferred_side = "R" if used_mirror else "L"

        rows.append(
            {
                "case_id": tag,
                "actual_side": side,
                "inferred_side": inferred_side,
                "side_correct": int(inferred_side == side),
                "volume_cm3": round(vcm3, 3),
                "mean_dist_mm": round(fit.mean_distance, 4),
                "rms_dist_mm": round(fit.rms_distance, 4),
                "hd95_mm": round(fit.hausdorff95, 4),
                "converged": int(fit.converged),
                "n_iter": fit.n_iter,
            }
        )
        flag = "" if inferred_side == side else "  (!side)"
        print(f"  [{i}/{len(lobes)}] {tag}: mean {fit.mean_distance:.2f} mm  "
              f"hd95 {fit.hausdorff95:.2f} mm  inferred {inferred_side}{flag}")

    if not rows:
        raise SystemExit(
            f"no lobes fitted: {len(excluded)} of {len(lobes)} excluded by QC "
            f"(check --volume-range; current [{vol_lo:g}, {vol_hi:g}] cm3)"
        )

    with open(out / "validation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mean_d = np.array([r["mean_dist_mm"] for r in rows], dtype=float)
    side_correct = np.array([r["side_correct"] for r in rows], dtype=float)
    summary = {
        "model_file": str(args.model),
        "n_modes_used": args.n_modes if args.n_modes is not None else model.n_modes,
        "n_lobes_found": len(lobes),
        "n_lobes_fitted": len(rows),
        "n_lobes_excluded_qc": len(excluded),
        "volume_range_cm3": [vol_lo, vol_hi],
        "excluded": excluded,
        "overall_mean_distance": summarise(mean_d),
        "per_side_mean_distance": {
            s: summarise(np.array([r["mean_dist_mm"] for r in rows if r["actual_side"] == s]))
            for s in ("L", "R")
        },
        "side_inference_accuracy": round(float(side_correct.mean()), 4),
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    o = summary["overall_mean_distance"]
    print(f"\nResults written to {out}/")
    print(f"  fitted {len(rows)}/{len(lobes)} lobes ({len(excluded)} excluded by QC)")
    print(f"  mean surface distance: median {o['median_mm']} mm, "
          f"mean {o['mean_mm']} mm, p95 {o['p95_mm']} mm")
    print(f"  side-inference accuracy (mirror-aware fit): "
          f"{summary['side_inference_accuracy']:.1%}")


if __name__ == "__main__":
    main()
