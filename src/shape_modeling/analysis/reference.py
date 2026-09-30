"""True medoid of a cohort, computed the way ShapeWorks intends -- but over *all* pairs.

``_groom_cohort`` picks the shape every other shape is rigidly aligned to with ShapeWorks'
``find_reference_image_index``. That calls ``MeshUtils::findReferenceMesh``, which for more
than 30 shapes compares only a random subset of 30 but then takes the argmin over *all* shapes;
the uncompared shapes keep a mean distance of 0, so one of them -- almost always cohort index 0
-- is returned instead of the medoid (ShapeWorks <= 6.7.0 and master as of 2026-09).

This module reproduces ShapeWorks' definition exactly, but over every pair, and splits the
O(n^2) work into SLURM-array-sized chunks:

1. :func:`prepare_reference_meshes` -- grooming step 1 (the shared
   :func:`~shape_modeling.analysis.correspondence._pregroom_binary`), ``computeDT(0.5)``,
   ``toMesh(0.5)`` -- the same meshes ``find_reference_image_index`` builds -- written to disk.
2. :func:`pairwise_chunk` -- for its share of the ``i < j`` pairs: rigid vtk ICP of mesh *i* onto
   mesh *j* (10 iterations, centroid start; ``Mesh.createTransform``), then the mean
   point-to-cell distance from the transformed *i* to *j* -- the per-pair value ShapeWorks uses.
3. :func:`reduce_pairwise` -- per-shape mean distance ``sum / (n - 1)``, argmin = medoid, plus
   a full ranking.

:func:`selfcheck` verifies step 2 against ShapeWorks itself on a small prefix (full pairwise,
``findReferenceMesh(meshes, 0)``) and shows the default call's result next to it.

Steps 1, 2 and the self-check need ShapeWorks; the chunking/reduction helpers are numpy-only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from shape_modeling.analysis.cohort import CohortEntry

    from shape_modeling.analysis.correspondence import ShapeWorksConfig

__all__ = [
    "case_ids_sha256",
    "chunk_bounds",
    "n_pairs",
    "pair_from_flat",
    "prepare_reference_meshes",
    "pairwise_chunk",
    "reduce_pairwise",
    "selfcheck",
]

ICP_ITERATIONS = 10  # what MeshUtils::findReferenceMesh uses
MESH_ISO = 0.5  # find_reference_image_index: img.toMesh(0.5) on each (pre-alignment) DT
DT_ISO = 0.5  # _groom_cohort: b.copy().computeDT(0.5)


# --------------------------------------------------------------------------------------------
# numpy-only helpers (unit-tested without ShapeWorks)
# --------------------------------------------------------------------------------------------


def n_pairs(n: int) -> int:
    """Number of unordered pairs ``i < j`` among ``n`` shapes."""
    return n * (n - 1) // 2


def pair_from_flat(k: int | np.ndarray, n: int) -> tuple[Any, Any]:
    """Map flat pair index(es) ``k`` to ``(i, j)``, ``i < j``, in ShapeWorks' enumeration order.

    ShapeWorks enumerates ``for i in range(n): for j in range(i + 1, n)``. Row ``i`` starts at
    ``start(i) = i*n - i*(i+1)/2``; invert with the quadratic formula, then fix off-by-ones from
    floating point.
    """
    k = np.asarray(k, dtype=np.int64)
    start = lambda i: i * n - (i * (i + 1)) // 2  # noqa: E731
    i = np.floor((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8.0 * k)) / 2).astype(np.int64)
    i = np.where(start(i + 1) <= k, i + 1, i)
    i = np.where(start(i) > k, i - 1, i)
    j = k - start(i) + i + 1
    return i, j


def chunk_bounds(n: int, chunk: int, n_chunks: int) -> tuple[int, int]:
    """Half-open flat-pair range ``[lo, hi)`` of ``chunk`` out of ``n_chunks`` (balanced)."""
    if not 0 <= chunk < n_chunks:
        raise ValueError(f"chunk {chunk} out of range for n_chunks={n_chunks}")
    total = n_pairs(n)
    return chunk * total // n_chunks, (chunk + 1) * total // n_chunks


def case_ids_sha256(case_ids: list[str]) -> str:
    """Fingerprint of an ordered cohort (sha256 of the newline-joined case ids)."""
    import hashlib

    return hashlib.sha256("\n".join(case_ids).encode()).hexdigest()


def means_from_pairs(n: int, i: np.ndarray, j: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Per-shape mean distance exactly as findReferenceMesh accumulates it (sum / (n - 1))."""
    sums = np.bincount(i, weights=d, minlength=n) + np.bincount(j, weights=d, minlength=n)
    return sums / (n - 1)


# --------------------------------------------------------------------------------------------
# ShapeWorks steps
# --------------------------------------------------------------------------------------------


def _sw() -> Any:
    from shape_modeling.analysis.correspondence import _require_shapeworks

    return _require_shapeworks()


def _mesh_path(mesh_dir: Path, idx: int) -> Path:
    return mesh_dir / f"{idx:05d}.vtp"


def prepare_reference_meshes(
    cohort: list[CohortEntry], config: ShapeWorksConfig, out_dir: str | Path
) -> Path:
    """Write the meshes ``find_reference_image_index`` would build, one ``.vtp`` per shape.

    Also writes ``manifest.json`` (case ids in cohort order, grooming parameters, ShapeWorks
    version) so later steps can check they work on the same cohort.
    """
    sw = _sw()
    out_dir = Path(out_dir)
    mesh_dir = out_dir / "meshes"
    seg_dir = out_dir / "segs"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    from shape_modeling.analysis.correspondence import _pregroom_binary

    n_points = []
    t0 = time.time()
    for idx, entry in enumerate(cohort):
        binary = _pregroom_binary(entry, config, seg_dir, sw)
        mesh = binary.copy().computeDT(DT_ISO).toMesh(MESH_ISO)
        mesh.write(str(_mesh_path(mesh_dir, idx)))
        n_points.append(int(mesh.numPoints()))
        if idx % 50 == 0:
            print(f"  prepared {idx + 1}/{len(cohort)} ({time.time() - t0:.0f}s)", flush=True)

    manifest = {
        "n": len(cohort),
        "case_ids": [e.case_id for e in cohort],
        "case_ids_sha256": case_ids_sha256([e.case_id for e in cohort]),
        "volume_cm3": [round(float(e.volume_cm3), 3) for e in cohort],
        "n_points": n_points,
        "iso_spacing": config.iso_spacing,
        "antialias_iterations": config.antialias_iterations,
        "dt_iso": DT_ISO,
        "mesh_iso": MESH_ISO,
        "icp_iterations": ICP_ITERATIONS,
        "shapeworks_version": getattr(sw, "__version__", "unknown"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(cohort)} meshes to {mesh_dir} "
          f"(points per mesh: median {int(np.median(n_points))}, max {max(n_points)})")
    return out_dir


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run prepare_reference_meshes first")
    return json.loads(path.read_text())


def _load_meshes(out_dir: Path, n: int, sw: Any) -> list[Any]:
    return [sw.Mesh(str(_mesh_path(out_dir / "meshes", i))) for i in range(n)]


def _pair_distance(src: Any, dst: Any, sw: Any) -> float:
    """One findReferenceMesh pair: rigid ICP src -> dst, then mean point-to-cell distance."""
    moved = src.copy()
    transform = moved.createTransform(dst, sw.Mesh.AlignmentType.Rigid, ICP_ITERATIONS)
    moved.applyTransform(transform)
    return float(np.mean(moved.distance(dst)[0]))


def pairwise_chunk(out_dir: str | Path, chunk: int, n_chunks: int) -> Path:
    """Compute this chunk's pair distances and write ``pairs/chunk_<k>.npz`` (i, j, d)."""
    sw = _sw()
    out_dir = Path(out_dir)
    n = int(_load_manifest(out_dir)["n"])
    lo, hi = chunk_bounds(n, chunk, n_chunks)
    i, j = pair_from_flat(np.arange(lo, hi), n)
    meshes = _load_meshes(out_dir, n, sw)

    d = np.empty(hi - lo, dtype=np.float64)
    t0 = time.time()
    for k, (a, b) in enumerate(zip(i.tolist(), j.tolist(), strict=True)):
        d[k] = _pair_distance(meshes[a], meshes[b], sw)
        if k and k % 500 == 0:
            rate = k / (time.time() - t0)
            print(f"  chunk {chunk}: {k}/{hi - lo} pairs ({rate:.1f}/s)", flush=True)

    pair_dir = out_dir / "pairs"
    pair_dir.mkdir(exist_ok=True)
    path = pair_dir / f"chunk_{chunk:04d}.npz"
    np.savez(path, i=i, j=j, d=d, lo=lo, hi=hi, n=n, n_chunks=n_chunks)
    print(f"chunk {chunk}/{n_chunks}: {hi - lo} pairs in {time.time() - t0:.0f}s -> {path}")
    return path


def reduce_pairwise(out_dir: str | Path) -> dict[str, Any]:
    """Combine all chunks -> per-shape means -> medoid; write ``reference.json`` + ranking CSV.

    Refuses to report a medoid unless every pair ``i < j`` is present exactly once.
    """
    out_dir = Path(out_dir)
    manifest = _load_manifest(out_dir)
    n, case_ids = int(manifest["n"]), list(manifest["case_ids"])

    files = sorted((out_dir / "pairs").glob("chunk_*.npz"))
    if not files:
        raise FileNotFoundError(f"no chunk files under {out_dir / 'pairs'}")
    parts = [np.load(f) for f in files]
    n_chunks = {int(p["n_chunks"]) for p in parts}
    if len(n_chunks) != 1 or len(files) != next(iter(n_chunks)):
        raise ValueError(f"expected {n_chunks} chunks, found {len(files)} files")
    i = np.concatenate([p["i"] for p in parts])
    j = np.concatenate([p["j"] for p in parts])
    d = np.concatenate([p["d"] for p in parts])
    flat = i * n - (i * (i + 1)) // 2 + (j - i - 1)
    if len(flat) != n_pairs(n) or len(np.unique(flat)) != n_pairs(n):
        raise ValueError(f"pair coverage incomplete: {len(np.unique(flat))} of {n_pairs(n)}")
    if not np.all(np.isfinite(d)):
        raise ValueError(f"{int(np.sum(~np.isfinite(d)))} non-finite pair distances")

    means = means_from_pairs(n, i, j, d)
    order = np.argsort(means, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    medoid = int(order[0])

    volumes = manifest.get("volume_cm3") or [""] * n
    with open(out_dir / "ranking.csv", "w") as fh:
        fh.write("rank,index,case_id,volume_cm3,mean_distance_mm\n")
        for r, idx in enumerate(order.tolist()):
            fh.write(f"{r},{idx},{case_ids[idx]},{volumes[idx]},{means[idx]:.6f}\n")

    result = {
        "medoid_index": medoid,
        "medoid_case_id": case_ids[medoid],
        "medoid_mean_distance_mm": float(means[medoid]),
        "index0_case_id": case_ids[0],
        "index0_rank": int(rank[0]),
        "index0_mean_distance_mm": float(means[0]),
        "medoid_volume_cm3": volumes[medoid],
        "index0_volume_cm3": volumes[0],
        "mean_distance_median_mm": float(np.median(means)),
        "top10": [{"index": int(k), "case_id": case_ids[k], "mean_distance_mm": float(means[k])}
                  for k in order[:10]],
        "n": n,
        "n_pairs": n_pairs(n),
        "shapeworks_version": manifest.get("shapeworks_version"),
    }
    (out_dir / "reference.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "top10"}, indent=1))
    print(f"-> medoid {case_ids[medoid]} (index {medoid}); index 0 ({case_ids[0]}) ranks "
          f"{int(rank[0])} of {n}")
    return result


def selfcheck(out_dir: str | Path, n_check: int = 40) -> dict[str, Any]:
    """Validate :func:`_pair_distance` against ShapeWorks on the first ``n_check`` meshes.

    * our full pairwise medoid vs ``findReferenceMesh(meshes, 0)`` (ShapeWorks, no subset) --
      must agree, otherwise the chunked computation does not reproduce ShapeWorks' definition;
    * ShapeWorks' default call ``findReferenceMesh(meshes)`` (random subset of 30) for
      comparison -- on real data this is the bug in action.

    Writes ``selfcheck.json``; raises if the first comparison disagrees.
    """
    sw = _sw()
    out_dir = Path(out_dir)
    n = min(int(_load_manifest(out_dir)["n"]), n_check)
    meshes = _load_meshes(out_dir, n, sw)

    i, j = pair_from_flat(np.arange(n_pairs(n)), n)
    pairs = zip(i.tolist(), j.tolist(), strict=True)
    d = np.array([_pair_distance(meshes[a], meshes[b], sw) for a, b in pairs])
    means = means_from_pairs(n, i, j, d)
    ours = int(np.argmin(means))

    sw_full = int(sw.MeshUtils.findReferenceMesh(meshes, 0))
    sw_default = int(sw.MeshUtils.findReferenceMesh(meshes))
    result = {
        "n_check": n,
        "ours_full_pairwise": ours,
        "shapeworks_full_pairwise": sw_full,
        "shapeworks_default": sw_default,
        # ties (e.g. two equally central shapes) count as agreement
        "agree": bool(ours == sw_full or np.isclose(means[ours], means[sw_full], rtol=1e-9)),
        "ours_mean_at_sw_full_mm": float(means[sw_full]),
        "ours_mean_at_ours_mm": float(means[ours]),
        "ours_mean_at_sw_default_mm": float(means[sw_default]),
        "sw_default_rank_in_prefix": int(np.argsort(np.argsort(means))[sw_default]),
        "shapeworks_version": getattr(sw, "__version__", "unknown"),
    }
    (out_dir / "selfcheck.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    if not result["agree"]:
        raise RuntimeError(
            "self-check FAILED: our pairwise medoid differs from ShapeWorks' full-pairwise one -- "
            "the chunked computation does not reproduce ShapeWorks; do not trust reduce_pairwise"
        )
    return result
