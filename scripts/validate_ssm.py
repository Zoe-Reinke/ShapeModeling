#!/usr/bin/env python
"""Held-out validation of an exported SSM: fit the model to unseen masks, report fit error.

This is the deployment-honest counterpart to the internal generalization curve. Internal
generalization (``evaluation.py``) measures leave-one-out reconstruction on the *training*
correspondence; this script instead takes a **held-out dataset the model never saw** -- a
different site, scanner, or annotation protocol -- and asks how well the exported model can
reconstruct each gland from its raw mask alone:

  1. load the exported model (``model.npz`` from ``build_prostate_ssm.py`` / ``export_model``);
  2. discover the held-out dataset(s) and take every case's whole-gland mask;
  3. clean each mask and extract its surface in physical mm (``clean_mask`` -> ``extract_surface``);
  4. fit the model to that **non-corresponded** surface (``ExportedModel.fit_surface``: ICP +
     similarity pose + mode-weight solve), with no correspondence step needed on the new shape;
  5. record the symmetric surface distance (mm) per case and summarise the distribution.

Crucially this needs **no ShapeWorks** -- ``fit_surface`` establishes its own pseudo-correspondence
by nearest-point matching, so a held-out mask goes straight from disk to a fit. Read the result
against the manual inter-observer floor of the held-out set (e.g. LUND-PROBE's 4-reader subset):
the model cannot fit closer than readers disagree.

Usage (from the repo root)::

    python scripts/validate_ssm.py --model ssm_output/model.npz \
        --data NCI-ISBI --data LUND-PROBE --out validation_output --n-modes 15

See ``python scripts/validate_ssm.py --help`` for all flags.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from shape_modeling.analysis import (
    CleanConfig,
    build_grouped_cohort,
    clean_mask,
    extract_surface,
    load_model,
)
from shape_modeling.io.reader import discover_cases
from shape_modeling.qc import QCConfig


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
        help="held-out dataset directory (repeat for several; source label = folder name)",
    )
    p.add_argument("--out", type=Path, default=Path("validation_output"), help="output folder")
    p.add_argument(
        "--n-modes",
        type=int,
        default=None,
        help="modes the fit may move (default: all the model has)",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian anti-alias sigma for clean_mask (voxels); 0 disables",
    )
    p.add_argument("--max-sd", type=float, default=3.0, help="clamp fitted weights to +/- this")
    p.add_argument("--n-restarts", type=int, default=4, help="ICP pose restarts per case")
    p.add_argument(
        "--volume-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(QCConfig().organ_volume_range_cm3),
        help="QC gate: drop masks whose whole-gland volume (cm3) falls outside [MIN, MAX] "
        "(default: prostate-plausible QCConfig range). Pass e.g. 0 1e9 to disable.",
    )
    p.add_argument(
        "--default-mri-type",
        type=str,
        default=None,
        help="default MRI sequence label passed to discover_cases (rarely needed for masks)",
    )
    return p.parse_args()


def summarise(distances: np.ndarray) -> dict:
    """Median / mean / 95th-percentile of a per-case distance array (mm)."""
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
          f"units={model.metadata.get('units')})")

    # Held-out cohort: every case with an organ mask, across all --data dirs.
    datasets = {
        d.name: discover_cases(d, default_mri_type=args.default_mri_type) for d in args.data
    }
    cohort = build_grouped_cohort(datasets)
    if not cohort:
        raise SystemExit("no held-out cases with an organ segmentation found")
    print(f"held-out cohort: {len(cohort)} cases from {', '.join(datasets)}")

    clean_cfg = CleanConfig(smooth_sigma=args.smooth_sigma)
    vol_lo, vol_hi = args.volume_range
    rows: list[dict] = []
    excluded: list[dict] = []
    for i, entry in enumerate(cohort, 1):
        # QC gate: an implausibly large/small mask is not a clean whole gland, and fitting it
        # would pollute the surface-distance distribution -- drop it (reported, never silent).
        if not (vol_lo <= entry.volume_cm3 <= vol_hi):
            reason = f"volume {entry.volume_cm3:.1f} cm3 outside [{vol_lo:g}, {vol_hi:g}]"
            excluded.append({"case_id": entry.case_id, "source": entry.source, "reason": reason})
            print(f"  [{i}/{len(cohort)}] {entry.case_id}: {reason} -- skipped (QC)")
            continue
        mask = clean_mask(entry.organ.array > 0, clean_cfg)
        if not mask.any():
            reason = "empty after cleaning"
            excluded.append({"case_id": entry.case_id, "source": entry.source, "reason": reason})
            print(f"  [{i}/{len(cohort)}] {entry.case_id}: {reason} -- skipped")
            continue
        target = extract_surface(mask, entry.organ).verts  # (N, 3) physical-mm point cloud
        fit = model.fit_surface(
            target, n_modes=args.n_modes, max_sd=args.max_sd, n_restarts=args.n_restarts
        )
        rows.append(
            {
                "case_id": entry.case_id,
                "source": entry.source,
                "is_cancer": int(entry.is_cancer),
                "volume_cm3": round(entry.volume_cm3, 3),
                "mean_dist_mm": round(fit.mean_distance, 4),
                "rms_dist_mm": round(fit.rms_distance, 4),
                "hd95_mm": round(fit.hausdorff95, 4),
                "converged": int(fit.converged),
                "n_iter": fit.n_iter,
            }
        )
        print(f"  [{i}/{len(cohort)}] {entry.case_id}: mean {fit.mean_distance:.2f} mm  "
              f"hd95 {fit.hausdorff95:.2f} mm")

    if not rows:
        raise SystemExit(
            f"no cases left to fit: {len(excluded)} of {len(cohort)} were excluded by QC "
            f"(check --volume-range; current [{vol_lo:g}, {vol_hi:g}] cm3)"
        )

    # Per-case CSV.
    with open(out / "validation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary.json -- distribution overall and per source.
    mean_d = np.array([r["mean_dist_mm"] for r in rows], dtype=float)
    sources = sorted({r["source"] for r in rows})
    summary = {
        "model_file": str(args.model),
        "n_modes_used": args.n_modes if args.n_modes is not None else model.n_modes,
        "n_cases_discovered": len(cohort),
        "n_cases_fitted": len(rows),
        "n_cases_excluded_qc": len(excluded),
        "volume_range_cm3": [vol_lo, vol_hi],
        "excluded": excluded,
        "overall_mean_distance": summarise(mean_d),
        "per_source_mean_distance": {
            s: summarise(np.array([r["mean_dist_mm"] for r in rows if r["source"] == s]))
            for s in sources
        },
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    o = summary["overall_mean_distance"]
    print(f"\nResults written to {out}/")
    print(f"  fitted {len(rows)}/{len(cohort)} cases ({len(excluded)} excluded by QC)")
    print(f"  mean surface distance: median {o['median_mm']} mm, "
          f"mean {o['mean_mm']} mm, p95 {o['p95_mm']} mm  (n={o['n']})")


if __name__ == "__main__":
    main()
