#!/usr/bin/env python3
"""SLURM array-task entry point used by ``template.sh`` (same pattern as ShapeModelReconstruction):

    python3 -u run_job.py $SLURM_ARRAY_TASK_ID "$EXPERIMENTS_FILE"

``EXPERIMENTS_FILE`` is a JSON list of job specs; the array task id picks one entry and its
``"task"`` key selects what runs. All other keys are passed through as keyword arguments. A spec
may set ``"chunk_from_task_id": <offset>`` -- then ``chunk = task_id - offset`` (so one spec
serves a whole array range instead of hundreds of near-identical entries). Submit e.g.

    sbatch --array=0 --export=ALL,EXPERIMENTS_FILE=experiments/reference_thyroid.json template.sh

Tasks (thyroid grooming-reference medoid, see shape_modeling.analysis.reference):
  reference_prepare    rebuild the training cohort, write the pre-alignment meshes (+ manifest)
  reference_selfcheck  our pair distance vs ShapeWorks on a prefix (must agree)
  reference_pairs      one chunk of the all-pairs ICP distances
  reference_reduce     combine chunks -> medoid, ranking.csv, reference.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _thyroid_cohort(data, exclude_file, holdout_frac, holdout_seed, expect_sha256=None):
    """The build_thyroid_ssm.py training cohort (same split, exclusions and lobe loading)."""
    from build_thyroid_ssm import build_cohort, parse_exclusions

    from shape_modeling.analysis.reference import case_ids_sha256

    exclude = parse_exclusions("", Path(exclude_file) if exclude_file else None)
    cohort, _ = build_cohort(Path(data), None, holdout_frac, holdout_seed, exclude)
    sha = case_ids_sha256([e.case_id for e in cohort])
    if expect_sha256 and sha != expect_sha256:
        raise SystemExit(
            f"cohort fingerprint {sha} != expected {expect_sha256}: this is not the cohort the "
            "model was built from (data path, exclusions or split differ) -- refusing to continue"
        )
    print(f"cohort: {len(cohort)} shapes, fingerprint {sha[:12]} "
          + ("(matches the model's cohort.csv)" if expect_sha256 else "(not checked)"), flush=True)
    return cohort


def _reference_prepare(out, data, exclude_file, holdout_frac=0.2, holdout_seed=0,
                       iso_spacing=1.5, expect_sha256=None):
    from shape_modeling.analysis.correspondence import ShapeWorksConfig
    from shape_modeling.analysis.reference import prepare_reference_meshes

    cohort = _thyroid_cohort(data, exclude_file, holdout_frac, holdout_seed, expect_sha256)
    # grooming parameters as in build_thyroid_ssm.py (ShapeWorksConfig defaults + iso 1.5 mm)
    prepare_reference_meshes(cohort, ShapeWorksConfig(iso_spacing=iso_spacing), out)


def _reference_selfcheck(out, n_check=40):
    from shape_modeling.analysis.reference import selfcheck

    selfcheck(out, n_check)


def _reference_pairs(out, n_chunks, chunk):
    from shape_modeling.analysis.reference import pairwise_chunk

    pairwise_chunk(out, chunk, n_chunks)


def _reference_reduce(out):
    from shape_modeling.analysis.reference import reduce_pairwise

    reduce_pairwise(out)


TASKS = {
    "reference_prepare": _reference_prepare,
    "reference_selfcheck": _reference_selfcheck,
    "reference_pairs": _reference_pairs,
    "reference_reduce": _reference_reduce,
}


def select_spec(specs: list[dict], task_id: int) -> dict:
    """Pick the spec for ``task_id``: an exact list index, or a ``chunk_from_task_id`` range."""
    for spec in specs:
        lo = spec.get("chunk_from_task_id")
        if lo is not None and lo <= task_id < lo + spec["n_chunks"]:
            spec = dict(spec)
            spec["chunk"] = task_id - spec.pop("chunk_from_task_id")
            return spec
    plain = [s for s in specs if "chunk_from_task_id" not in s]
    if not 0 <= task_id < len(plain):
        raise SystemExit(f"no spec for task id {task_id}")
    return dict(plain[task_id])


def main(argv):
    if len(argv) != 3:
        raise SystemExit(f"usage: {argv[0]} TASK_ID EXPERIMENTS_FILE")
    task_id, path = int(argv[1]), argv[2]
    with open(path) as f:
        specs = json.load(f)
    spec = select_spec(specs, task_id)
    task = spec.pop("task")
    spec.pop("comment", None)
    print(f"run_job: task {task_id} -> {task} {spec}", flush=True)
    TASKS[task](**spec)


if __name__ == "__main__":
    main(sys.argv)
