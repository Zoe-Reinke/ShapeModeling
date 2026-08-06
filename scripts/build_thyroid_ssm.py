#!/usr/bin/env python
"""Final-run thyroid lobe SSM build: a pooled, side-agnostic shape prior for 3D-US reconstruction.

Production counterpart to ``notebooks/explore_thyroid_lobes.ipynb`` (copy-and-edited from
``build_prostate_ssm.py`` per the "one script per organ" convention in CLAUDE.md). It builds a
**single, non-side-specific** lobe model: left and right lobes are pooled into one correspondence
+ PCA, with the **right lobes mirrored** to the left's handedness so both sides share a chiral
frame (Procrustes has no reflection). Because both sides are in the training set, the model's modes
span left- and right-typical shapes -- so one model reconstructs either lobe, and the deployment
fit never needs a side label (see ``validate_thyroid_ssm.py``, which fits both chiralities).

Pipeline:
  1. load every ``<case>/segmentations/thyroid_{left,right}.nii.gz`` mask directly with SimpleITK
     (TotalSegmentator filenames don't match the reader's tokens), crop each to its ROI, recentre
     its centre-of-mass to the world origin, and mirror the right lobes -> one CohortEntry per lobe;
  2. groom + run ShapeWorks particle correspondence (cached in ``--work-dir``, self-cleaning on a
     degenerate spread ratio);
  3. GPA (Procrustes; ``scale=True`` by default -> unit-size normalized model, the flexible prior
     for a 7-DOF similarity fit at deployment) -> PCA;
  4. compute the three SSM quality curves (compactness / generalization / specificity);
  5. export the model (``model.npz``) and write all numbers to ``--out``.

ShapeWorks is **not** pip-installable -- run this from the ShapeWorks conda environment. A cached
``particles.npz`` lets re-runs skip the optimizer.

Usage (from the repo root, inside the ShapeWorks env)::

    python scripts/build_thyroid_ssm.py \
        --data Thyroid/Totalsegmentator_dataset_v201 \
        --num-particles 512 --out thyroid_ssm_output

QC-failed subjects (mis-split lobes, implausibly small masks) are dropped with
``--exclude-subjects`` / ``--exclude-file``. Because the cache is keyed on the case-id list, a
changed exclusion set re-optimizes from scratch **and overwrites the cache in place** -- pass a
fresh ``--work-dir`` (and ``--out``) so the previous model's particles survive for comparison::

    python scripts/build_thyroid_ssm.py --holdout-frac 0.2 --num-particles 512 \
        --exclude-file thyroid_excluded_subjects.txt \
        --work-dir shapeworks_work_thyroid_final_512p_excl \
        --out thyroid_ssm_output_excl

See ``python scripts/build_thyroid_ssm.py --help`` for all flags.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from shape_modeling.analysis import (
    CohortEntry,
    ShapeWorksConfig,
    build_ssm,
    evaluate_ssm,
    export_model,
    gland_volume_cm3,
    gpa,
    hull_surface,
    run_shapeworks,
)
from shape_modeling.analysis.correspondence import ParticleSystem, particle_spread_ratio
from shape_modeling.io.reader import LoadedVolume, VolumeType

# Grooming/mirroring constants -- the values validated in the exploratory notebook.
MIRROR_AXIS = 2       # SimpleITK-array axis (z, y, x); 2 = x = left-right
ISO_SPACING = 1.5     # mm; grooming resample
ROI_MARGIN = 8        # voxels of background kept when cropping a lobe out of the full CT grid
MIN_VOLUME_CM3 = 0.3  # drop degenerate / empty lobe masks below this


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("Thyroid/Totalsegmentator_dataset_v201"),
        help="TotalSegmentator-style root: <case>/segmentations/thyroid_{left,right}.nii.gz",
    )
    p.add_argument("--num-particles", type=int, default=512, help="ShapeWorks particles per lobe")
    p.add_argument(
        "--iterations", type=int, default=1000, help="ShapeWorks optimization iterations"
    )
    p.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="cap subjects (each gives 2 lobes); default all",
    )
    p.add_argument(
        "--holdout-frac",
        type=float,
        default=0.0,
        help="reserve this fraction of subjects (both lobes) for held-out validation; their case "
        "ids are written to <out>/heldout.txt and excluded from training. Default 0 (train on all)",
    )
    p.add_argument(
        "--holdout-seed", type=int, default=0, help="seed for the deterministic held-out split"
    )
    p.add_argument(
        "--exclude-subjects",
        default="",
        help="comma-separated case ids to drop entirely (QC failures: mis-split lobes, "
        "implausibly small masks). Bare numbers are accepted and zero-padded (303 -> s0303)",
    )
    p.add_argument(
        "--exclude-file",
        type=Path,
        default=None,
        help="file with one case id per line to drop (merged with --exclude-subjects; "
        "'#' comments and blank lines ignored)",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="ShapeWorks cache dir (default: shapeworks_work_thyroid_final_<N>p)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("thyroid_ssm_output"), help="output folder for results"
    )
    p.add_argument(
        "--no-scale",
        action="store_true",
        help="GPA without scale normalization (keeps size; exported model is in mm)",
    )
    p.add_argument("--seed", type=int, default=0, help="seed for the stochastic specificity metric")
    return p.parse_args()


def load_lobe(path: Path, case_id: str, mirror: bool) -> CohortEntry | None:
    """Read one lobe mask, crop to its ROI, recentre COM to origin, optionally mirror.

    Mirrors ``load_lobe`` in the exploratory notebook. Cropping shrinks the full neck/chest CT
    grid (~10^7 mostly-background voxels) ~100-700x; recentring co-locates every lobe at the
    world origin so ShapeWorks' COM + ICP grooming starts near-aligned (otherwise a lobe at its
    own far CT coordinate lands outside the reference frame and grooms to empty). Mirroring
    reflects the right lobe onto the left's handedness. None if empty / below MIN_VOLUME_CM3.
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    if not (arr > 0).any():
        return None
    zi, yi, xi = np.nonzero(arr > 0)
    m = ROI_MARGIN
    z0, z1 = max(0, int(zi.min()) - m), min(arr.shape[0], int(zi.max()) + 1 + m)
    y0, y1 = max(0, int(yi.min()) - m), min(arr.shape[1], int(yi.max()) + 1 + m)
    x0, x1 = max(0, int(xi.min()) - m), min(arr.shape[2], int(xi.max()) + 1 + m)
    img = img[x0:x1, y0:y1, z0:z1]  # SimpleITK indexes (x, y, z); origin is updated
    arr = sitk.GetArrayFromImage(img)
    if mirror:
        arr = np.flip(arr, axis=MIRROR_AXIS).copy()
        flipped = sitk.GetImageFromArray(arr)
        flipped.CopyInformation(img)
        img = flipped
    com_idx = np.nonzero(arr > 0)
    com_xyz = [float(np.mean(com_idx[2])), float(np.mean(com_idx[1])), float(np.mean(com_idx[0]))]
    com_phys = np.array(img.TransformContinuousIndexToPhysicalPoint(com_xyz))
    img.SetOrigin(tuple(np.array(img.GetOrigin()) - com_phys))
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
    if gland_volume_cm3(vol) < MIN_VOLUME_CM3:
        return None
    return CohortEntry(
        case_id=case_id,
        source="TotalSegmentator",
        volume_cm3=gland_volume_cm3(vol),
        organ=vol,
        is_cancer=mirror,  # repurposed: True = right lobe (mirrored)
    )


def parse_exclusions(spec: str, path: Path | None) -> set[str]:
    """Normalise the ``--exclude-subjects`` / ``--exclude-file`` ids to ``sNNNN`` case-dir names.

    Bare integers are zero-padded to the TotalSegmentator convention (``303`` -> ``s0303``) so a
    list copy-pasted from a QC review works without hand-editing.
    """
    tokens = [t.strip() for t in spec.split(",")]
    if path is not None:
        if not path.is_file():
            raise SystemExit(f"--exclude-file not found: {path}")
        tokens += [ln.split("#", 1)[0].strip() for ln in path.read_text().splitlines()]
    ids = set()
    for t in tokens:
        if not t:
            continue
        ids.add(f"s{int(t):04d}" if t.isdigit() else t)
    return ids


def build_cohort(
    data_dir: Path,
    max_subjects: int | None,
    holdout_frac: float,
    holdout_seed: int,
    exclude: set[str] | None = None,
) -> tuple[list[CohortEntry], list[str]]:
    """Assemble the pooled both-sides lobe cohort; optionally reserve a held-out subject split.

    Returns ``(cohort, heldout_ids)``. The split is at the **subject** level (both of a subject's
    lobes go to the same side of the split, so a subject is never in both train and test) and is
    deterministic given ``holdout_seed``. Held-out subjects are excluded from ``cohort`` entirely.

    ``exclude`` drops QC-failed subjects (both lobes) **after** the held-out split is drawn, not
    before: the split is a permutation of the candidate list, so removing subjects up front would
    reshuffle every assignment and invalidate a previously validated ``heldout.txt``. Excluding
    afterwards leaves the split membership identical and only removes the bad subjects from
    whichever side they landed on.
    """
    if not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    # Candidate subjects: those with both lobe files present (integrity is checked at load time).
    candidates = [
        cdir
        for cdir in sorted(p for p in data_dir.iterdir() if p.is_dir())
        if (cdir / "segmentations" / "thyroid_left.nii.gz").exists()
        and (cdir / "segmentations" / "thyroid_right.nii.gz").exists()
    ]
    if not candidates:
        raise SystemExit(f"no thyroid lobe pairs found under {data_dir}")

    # Deterministic subject-level held-out split.
    heldout_ids: list[str] = []
    train_dirs = candidates
    if holdout_frac > 0:
        if not 0 < holdout_frac < 1:
            raise SystemExit(f"--holdout-frac must be in (0, 1), got {holdout_frac}")
        order = np.random.default_rng(holdout_seed).permutation(len(candidates))
        n_hold = round(holdout_frac * len(candidates))
        hold = {int(i) for i in order[:n_hold]}
        heldout_ids = sorted(candidates[i].name for i in hold)
        train_dirs = [c for i, c in enumerate(candidates) if i not in hold]

    if exclude:
        unknown = exclude - {c.name for c in candidates}
        if unknown:
            raise SystemExit(f"--exclude ids not present under {data_dir}: {sorted(unknown)}")
        n_before = len(train_dirs)
        train_dirs = [c for c in train_dirs if c.name not in exclude]
        dropped_train = n_before - len(train_dirs)
        dropped_hold = sum(1 for h in heldout_ids if h in exclude)
        heldout_ids = [h for h in heldout_ids if h not in exclude]
        print(f"excluded {len(exclude)} QC-failed subjects: {dropped_train} from training"
              + (f", {dropped_hold} from the held-out split" if dropped_hold else ""))

    cohort: list[CohortEntry] = []
    n_subjects = skipped = 0
    for cdir in train_dirs:
        if max_subjects is not None and n_subjects >= max_subjects:
            break
        seg = cdir / "segmentations"
        left = load_lobe(seg / "thyroid_left.nii.gz", f"{cdir.name}_L", mirror=False)
        right = load_lobe(seg / "thyroid_right.nii.gz", f"{cdir.name}_R", mirror=True)
        if left is None or right is None:  # keep only complete pairs so the two sides balance
            skipped += 1
            continue
        cohort.extend([left, right])
        n_subjects += 1
    if not cohort:
        raise SystemExit(f"no usable thyroid lobe pairs under {data_dir} after the split")
    print(f"Cohort: {len(cohort)} lobes from {n_subjects} subjects "
          f"({sum(not e.is_cancer for e in cohort)} left / {sum(e.is_cancer for e in cohort)} "
          f"right); {skipped} subjects skipped (a lobe was empty/degenerate)"
          + (f"; {len(heldout_ids)} subjects held out" if heldout_ids else ""))
    return cohort, heldout_ids


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir or Path(f"shapeworks_work_thyroid_final_{args.num_particles}p")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    scale = not args.no_scale

    # 1. Cohort -----------------------------------------------------------------------------
    exclude = parse_exclusions(args.exclude_subjects, args.exclude_file)
    cohort, heldout_ids = build_cohort(
        args.data, args.max_subjects, args.holdout_frac, args.holdout_seed, exclude
    )
    if exclude:
        (out / "excluded.txt").write_text("\n".join(sorted(exclude)) + "\n")
    if heldout_ids:
        (out / "heldout.txt").write_text("\n".join(heldout_ids) + "\n")
        print(f"reserved {len(heldout_ids)} subjects -> {out / 'heldout.txt'} "
              f"(pass to validate_thyroid_ssm.py --holdout-list)")

    # 2. Correspondence (ShapeWorks: groom + optimize, cached) ------------------------------
    sw_cfg = ShapeWorksConfig(
        num_particles=args.num_particles,
        iterations=args.iterations,
        iso_spacing=ISO_SPACING,
        work_dir=work_dir,
        use_cache=True,
    )
    # Self-clean a stale/degenerate cache so grooming re-optimizes; a healthy cache is kept.
    cache = work_dir / "particles.npz"
    if cache.exists() and sw_cfg.max_spread_ratio is not None:
        old = ParticleSystem.load(cache)
        if old.case_ids == [e.case_id for e in cohort] and (
            particle_spread_ratio(old, cohort) > sw_cfg.max_spread_ratio
        ):
            cache.unlink()
            print("dropped degenerate cache -> will re-groom + re-optimize")

    system = run_shapeworks(cohort, sw_cfg)
    assert system.case_ids == [e.case_id for e in cohort]  # rows align with metadata
    spread = particle_spread_ratio(system, cohort)
    print(f"particles: {system.particles.shape}  spread ratio: {spread:.2f} (healthy ~1.0-1.3)")

    # 3. GPA -> PCA (no ComBat: TotalSegmentator carries no per-lobe site label) -------------
    aligned, _ = gpa(system.particles, scale=scale)
    model = build_ssm(aligned)
    print(f"SSM: {model.n_modes} modes over {model.n_points} points, {len(cohort)} lobes")

    # 4. SSM quality metrics (on world particles in mm; Studio convention) ------------------
    evaluation = evaluate_ssm(system.particles, seed=args.seed, label="thyroid-pooled")

    # 5. Export model + write all numbers ---------------------------------------------------
    n_right = int(sum(e.is_cancer for e in cohort))
    units = "normalized" if scale else "mm"
    metadata = {
        "organ": "thyroid_lobe",
        "side_agnostic": True,
        "mirror_axis": MIRROR_AXIS,
        "chirality_frame": "left (right lobes mirrored in)",
        "cohort_size": len(cohort),
        "n_left": len(cohort) - n_right,
        "n_right": n_right,
        "num_particles": args.num_particles,
        "iso_spacing_mm": ISO_SPACING,
        "gpa_scale": scale,
        "spread_ratio": round(spread, 3),
        "holdout_frac": args.holdout_frac,
        "n_heldout_subjects": len(heldout_ids),
        "n_excluded_subjects": len(exclude),
        "excluded_subjects": sorted(exclude),
    }
    faces = hull_surface(model.mean).faces  # quick convex-hull connectivity for viewing
    model_path = export_model(model, out / "model.npz", faces=faces, units=units, metadata=metadata)
    print(f"wrote {model_path}")

    # cohort.csv -- one row per modelled lobe
    with open(out / "cohort.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "side", "volume_cm3"])
        for e in cohort:
            w.writerow([e.case_id, "R" if e.is_cancer else "L", f"{e.volume_cm3:.3f}"])

    # eigenvalues.csv -- the PCA spectrum
    cum = np.cumsum(model.explained_variance_ratio)
    with open(out / "eigenvalues.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "eigenvalue", "explained_variance_ratio", "cumulative_ratio"])
        for i in range(model.n_modes):
            w.writerow([i + 1, model.eigenvalues[i], model.explained_variance_ratio[i], cum[i]])

    # evaluation.csv -- the three SSM quality curves vs number of modes
    evaluation.to_frame().to_csv(out / "evaluation.csv", index=False)

    # summary.json -- headline numbers
    summary = {
        **metadata,
        "n_modes": model.n_modes,
        "n_points": model.n_points,
        "modes_for_90pct_variance": evaluation.modes_for_variance(0.90),
        "modes_for_95pct_variance": evaluation.modes_for_variance(0.95),
        "modes_for_99pct_variance": evaluation.modes_for_variance(0.99),
        "generalization_mm_at_max_modes": float(evaluation.generalization[-1]),
        "specificity_mm_at_max_modes": float(evaluation.specificity[-1]),
        "model_file": model_path.name,
        "units": units,
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nResults written to {out}/")
    print(f"  modes for 90/95/99% variance: "
          f"{summary['modes_for_90pct_variance']}/"
          f"{summary['modes_for_95pct_variance']}/"
          f"{summary['modes_for_99pct_variance']}")
    print(f"  generalization @ max modes: {summary['generalization_mm_at_max_modes']:.3f} mm")
    print(f"  specificity @ max modes:    {summary['specificity_mm_at_max_modes']:.3f} mm")


if __name__ == "__main__":
    main()
