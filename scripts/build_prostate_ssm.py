#!/usr/bin/env python
"""Final-run prostate SSM build: discover -> groom + correspond -> GPA -> (ComBat) -> PCA -> export.

This is the *production* counterpart to the exploratory notebooks: it keeps only what the
final Statistical Shape Model needs and writes every result number to an output folder. The
pipeline is:

  1. discover all datasets (SAML + Prostate158 + PI-CAI if present), build the whole-gland
     cohort (healthy + cancer), drop SAML's I2CVB site, relabel SAML by acquisition centre,
     and apply PI-CAI's csPCa labels;
  2. groom + run ShapeWorks particle correspondence (cached in ``--work-dir``);
  3. GPA (Procrustes) -> optional ComBat harmonization on acquisition centre -> re-GPA -> PCA;
  4. compute the three SSM quality curves (compactness / generalization / specificity);
  5. export the model (``model.npz``) and write all numbers to ``--out``.

ShapeWorks is **not** pip-installable -- run this from the ShapeWorks conda environment
(``import shapeworks`` + the ``shapeworks`` CLI on PATH). Everything after correspondence
works without it, and a cached ``particles.npz`` lets re-runs skip the optimizer.

Usage (from the repo root, inside the ShapeWorks env)::

    python scripts/build_prostate_ssm.py \
        --saml SAML --prostate158 Prostate158 --picai PICAI \
        --num-particles 1024 --out ssm_output

See ``python scripts/build_prostate_ssm.py --help`` for all flags.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from shape_modeling.analysis import (
    ShapeWorksConfig,
    build_grouped_cohort,
    build_ssm,
    centroid_size,
    combat_harmonize,
    evaluate_ssm,
    export_model,
    gpa,
    reconstruct_surface,
    run_shapeworks,
)
from shape_modeling.analysis.correspondence import ParticleSystem, particle_spread_ratio
from shape_modeling.io.reader import discover_cases

# SAML site whose masks are planar-truncated -- excluded everywhere (see CLAUDE.md / notebooks).
EXCLUDE_SITES = {"I2CVB"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--saml", type=Path, default=Path("SAML"), help="SAML multi-centre directory")
    p.add_argument(
        "--prostate158", type=Path, default=Path("Prostate158"), help="Prostate158 directory"
    )
    p.add_argument(
        "--picai",
        type=Path,
        default=Path("PICAI"),
        help="PI-CAI directory (skipped if absent; expects labels.csv from fetch_picai.py)",
    )
    p.add_argument(
        "--num-particles", type=int, default=1024, help="ShapeWorks particles per shape"
    )
    p.add_argument(
        "--iterations", type=int, default=1000, help="ShapeWorks optimization iterations"
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="ShapeWorks grooming/optimize cache dir (default: shapeworks_work_final_<N>p)",
    )
    p.add_argument("--out", type=Path, default=Path("ssm_output"), help="output folder for results")
    p.add_argument(
        "--no-scale",
        action="store_true",
        help="GPA without scale normalization (keeps size; exported model is in mm)",
    )
    p.add_argument(
        "--no-harmonize",
        action="store_true",
        help="skip ComBat harmonization on acquisition centre (single-centre or raw model)",
    )
    p.add_argument("--seed", type=int, default=0, help="seed for the stochastic specificity metric")
    return p.parse_args()


def build_cohort(args: argparse.Namespace) -> list:
    """Discover the datasets and assemble the labelled whole-gland cohort."""
    datasets: dict[str, object] = {}
    if args.saml.is_dir():
        datasets["SAML"] = discover_cases(args.saml, default_mri_type="T2W")
    if args.prostate158.is_dir():
        datasets["Prostate158"] = discover_cases(args.prostate158, default_mri_type="T2W")
    picai_dir = args.picai if args.picai.is_dir() else None
    if picai_dir is not None:
        datasets["PICAI"] = discover_cases(picai_dir)  # masks only -- no MRI, no default type
    if not datasets:
        raise SystemExit("no input datasets found; pass --saml / --prostate158 / --picai")

    cohort = build_grouped_cohort(datasets)

    # Drop the planar-truncated SAML I2CVB site.
    cohort = [e for e in cohort if e.case_id.split("/")[0] not in EXCLUDE_SITES]

    # Relabel each SAML entry's source to its acquisition centre (the ComBat batch unit).
    for e in cohort:
        if e.source == "SAML":
            e.source = "SAML:" + e.case_id.split("/")[0]

    # PI-CAI's AI folder carries no cancer status -> set is_cancer from the marksheet sidecar.
    if picai_dir is not None and (picai_dir / "labels.csv").exists():
        with open(picai_dir / "labels.csv") as fh:
            labels = {r["case_id"]: r for r in csv.DictReader(fh)}
        for e in cohort:
            if e.source == "PICAI":
                row = labels.get(e.case_id)
                if row is not None:
                    e.is_cancer = row["is_cancer"] == "1"
    return cohort


def centre_of(source: str) -> str:
    """ComBat batch label: the acquisition centre (Prostate158, PICAI, or SAML's site)."""
    return source.split(":")[-1]


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir or Path(f"shapeworks_work_final_{args.num_particles}p")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    scale = not args.no_scale
    harmonize = not args.no_harmonize

    # 1. Cohort -----------------------------------------------------------------------------
    cohort = build_cohort(args)
    sources = np.array([e.source for e in cohort])
    centres = np.array([centre_of(s) for s in sources])
    is_cancer = np.array([e.is_cancer for e in cohort])
    n_cancer = int(is_cancer.sum())
    print(f"Cohort: {len(cohort)} cases ({n_cancer} cancer, {len(cohort) - n_cancer} healthy)")
    for s, n in sorted(Counter(sources).items()):
        nc = sum(e.is_cancer for e in cohort if e.source == s)
        print(f"  {s:<18}{n:>4}  ({nc} cancer)")

    # 2. Correspondence (ShapeWorks: groom + optimize, cached) ------------------------------
    sw_cfg = ShapeWorksConfig(
        num_particles=args.num_particles,
        iterations=args.iterations,
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

    # 3. GPA -> (ComBat on centre) -> re-GPA -> PCA -----------------------------------------
    raw_size = centroid_size(system.particles)  # pre-GPA size (mm) -- protected ComBat covariate
    aligned, _ = gpa(system.particles, scale=scale)
    if harmonize and np.unique(centres).size >= 2:
        harm = combat_harmonize(aligned, centres, raw_size, covariate_names=["centroid_size"])
        aligned, _ = gpa(harm, scale=scale)
        print(f"ComBat-harmonized on centre ({np.unique(centres).size} batches)")
    elif harmonize:
        print("only one acquisition centre -> skipping ComBat")
        harmonize = False

    model = build_ssm(aligned)
    print(f"SSM: {model.n_modes} modes over {model.n_points} points, {len(cohort)} subjects")

    # 4. SSM quality metrics (on world particles in mm; Studio convention) ------------------
    evaluation = evaluate_ssm(system.particles, seed=args.seed, label="final")

    # 5. Export model + write all numbers ---------------------------------------------------
    units = "normalized" if scale else "mm"
    metadata = {
        "cohort_size": len(cohort),
        "n_cancer": n_cancer,
        "n_healthy": len(cohort) - n_cancer,
        "sources": dict(sorted(Counter(sources).items())),
        "num_particles": args.num_particles,
        "gpa_scale": scale,
        "combat_harmonized_on_centre": harmonize,
        "spread_ratio": round(spread, 3),
    }
    faces = reconstruct_surface(model.mean).faces  # real (concavity-following) topology
    model_path = export_model(model, out / "model.npz", faces=faces, units=units, metadata=metadata)
    print(f"wrote {model_path}")

    # cohort.csv -- one row per modelled subject
    with open(out / "cohort.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "source", "centre", "volume_cm3", "is_cancer"])
        for e in cohort:
            w.writerow(
                [e.case_id, e.source, centre_of(e.source), f"{e.volume_cm3:.3f}", int(e.is_cancer)]
            )

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
