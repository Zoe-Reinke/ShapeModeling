"""Particle-based correspondence across subjects via ShapeWorks.

ShapeWorks places ``M`` corresponding particles on every shape so that particle ``j`` marks
the same anatomical location on each subject -- the correspondence PCA needs. The flow is:
groom each whole-gland segmentation (antialias -> isotropic resample -> centre-of-mass +
rigid alignment to a reference -> crop -> signed distance transform), run the particle
optimizer over the groomed shapes, then read the per-subject ``*_world.particles`` into an
``(S, M, 3)`` array.

**ShapeWorks is not pip-installable.** Install it from the official distribution and run this
module from its environment (so ``import shapeworks`` works and the ``shapeworks`` CLI is on
PATH). Everything else in :mod:`shape_modeling.analysis` works without it; the import here is
deferred to call time so this module loads regardless.

The optimizer is expensive, so :func:`run_shapeworks` caches its result in ``work_dir`` and
reloads it when the cohort is unchanged. :func:`load_particles` reads particle files produced
by any ShapeWorks run (e.g. ShapeWorks Studio) without needing the package installed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import SimpleITK as sitk

if TYPE_CHECKING:
    from shape_modeling.analysis.cohort import CohortEntry

__all__ = [
    "ParticleSystem",
    "ShapeWorksConfig",
    "load_particles",
    "particle_spread_ratio",
    "run_shapeworks",
]


@dataclass
class ShapeWorksConfig:
    """ShapeWorks grooming + optimization parameters (exploratory defaults)."""

    num_particles: int = 512
    iterations: int = 1000
    relative_weighting: float = 1.0
    starting_regularization: float = 1000.0
    ending_regularization: float = 10.0
    iso_spacing: float = 1.0  # isotropic resample (mm) during grooming
    antialias_iterations: int = 30
    pad_voxels: int = 20  # background margin (voxels): added before the rigid transform so a
    # rotated gland is not clipped, and again around each cropped mask so the DT has room on every
    # face (a gland touching the scan's field-of-view edge still gets queried safely)
    icp_iterations: int = 200  # iterations for the ICP rigid alignment to the reference shape
    align_iso_value: float = 0.0  # DT level-set used as the surface for ICP (a DT's surface is 0)
    blur_sigma: float = 1.3  # Gaussian smoothing (mm) of the DT; gentler gradients + smoother
    # surfaces (ShapeWorks' recommended default for segmentations)
    # How far (in voxels) from the zero level-set the optimizer trusts the distance transform.
    # Must exceed the deepest interior distance (a prostate half-thickness is ~20 mm, so the DT
    # reaches ~+20 inside): ShapeWorks' default ~4 is too small and samples taken in the interior at
    # init abort with "outside the narrow band". But an effectively-infinite value (1e10) makes the
    # optimizer hold the *entire* DT of every subject in memory and the OS kills it (OOM, exit -9).
    # 30 covers the interior with margin while staying bounded; raise if you still see band errors,
    # lower (or drop num_particles/subject count) if you run out of memory.
    narrow_band: float = 30.0
    work_dir: Path = field(default_factory=lambda: Path("shapeworks_work"))
    use_cache: bool = True
    # Sanity gate on the particles (fresh or cached): the median over subjects of
    # (particle-cloud extent / organ-mask extent) must not exceed this. A healthy correspondence
    # hugs the organ (ratio ~1.0-1.3); a degenerate distance transform scatters particles across
    # the padded image (observed ~3.2), so this rejects e.g. a cache built before a grooming fix.
    # Set to None to disable the check.
    max_spread_ratio: float | None = 1.75
    # Index (into the cohort) of the grooming reference shape. None = ShapeWorks'
    # find_reference_image_index, which for > 30 shapes does NOT return the medoid (ShapeWorks
    # <= 6.7.0 / master as of 2026-09: MeshUtils::findReferenceMesh takes the argmin over all
    # shapes although only a random subset of 30 was compared, so an uncompared shape -- almost
    # always index 0 -- wins). Compute the true medoid with shape_modeling.analysis.reference and
    # pass its index here. Note: changing the reference changes grooming, so a cached
    # particles.npz from a run with a different reference no longer matches -- use a new work_dir.
    reference_index: int | None = None


@dataclass
class ParticleSystem:
    """Corresponded particles for a cohort: ``particles[s, j]`` is point ``j`` of subject ``s``."""

    case_ids: list[str]
    particles: np.ndarray  # (S, M, 3)

    def save(self, path: str | Path) -> None:
        """Save to a ``.npz`` (case ids + particle array)."""
        np.savez(path, case_ids=np.array(self.case_ids, dtype=object), particles=self.particles)

    @classmethod
    def load(cls, path: str | Path) -> ParticleSystem:
        data = np.load(path, allow_pickle=True)
        return cls(case_ids=list(data["case_ids"]), particles=data["particles"])


def _require_shapeworks() -> object:
    """Import and return the ``shapeworks`` module, with an actionable error if missing."""
    try:
        import shapeworks as sw
    except ImportError as exc:  # pragma: no cover - exercised only without ShapeWorks
        raise ImportError(
            "ShapeWorks is required for particle correspondence but is not importable. "
            "It is not pip-installable: install the official ShapeWorks distribution and "
            "run from its environment (`conda activate shapeworks`). "
            "See https://sciinstitute.github.io/ShapeWorks/."
        ) from exc
    return sw


def _sanitize(case_id: str) -> str:
    """Make a case id safe as a filename stem (namespaced ids contain ``/``)."""
    return case_id.replace("/", "__").replace("\\", "__")


def load_particles(
    directory: str | Path,
    case_ids: list[str],
    *,
    suffix: str = "_world.particles",
) -> ParticleSystem:
    """Load ``<sanitized_id><suffix>`` particle files (one per case) into a :class:`ParticleSystem`.

    Each file is whitespace-delimited ``x y z`` rows. All files must share the same particle
    count. Works without ShapeWorks installed.
    """
    directory = Path(directory)
    arrays: list[np.ndarray] = []
    for cid in case_ids:
        path = directory / f"{_sanitize(cid)}{suffix}"
        if not path.exists():
            raise FileNotFoundError(f"missing particle file for case {cid!r}: {path}")
        arrays.append(np.loadtxt(path, dtype=np.float64).reshape(-1, 3))
    counts = {a.shape[0] for a in arrays}
    if len(counts) != 1:
        raise ValueError(f"particle files disagree on point count: {sorted(counts)}")
    return ParticleSystem(case_ids=list(case_ids), particles=np.stack(arrays))


def particle_spread_ratio(system: ParticleSystem, cohort: list[CohortEntry]) -> float:
    """Median over subjects of ``(particle-cloud extent) / (organ-mask extent)``.

    A scale-free health check on the correspondence. For each subject it divides the largest
    bounding-box side of the particle cloud by the largest physical side of the organ mask;
    a correspondence that hugs the organ gives ~1.0--1.3, while particles scattered across the
    padded image (e.g. from a degenerate distance transform) give a much larger ratio. Subjects
    whose organ pixels are not loaded, or with an empty mask, are skipped; returns ``nan`` if no
    subject can be measured.
    """
    ratios: list[float] = []
    for entry, pts in zip(cohort, system.particles, strict=False):
        if entry.organ.array is None:
            continue
        voxels = np.argwhere(np.asarray(entry.organ.array) > 0)
        if voxels.size == 0:
            continue
        spacing_zyx = np.asarray(entry.organ.spacing, dtype=np.float64)[::-1]  # (x,y,z)->(z,y,x)
        organ_extent = float(((voxels.max(0) - voxels.min(0)) * spacing_zyx).max())
        if organ_extent <= 0:
            continue
        ratios.append(float((pts.max(0) - pts.min(0)).max()) / organ_extent)
    return float(np.median(ratios)) if ratios else float("nan")


def _check_particles(
    system: ParticleSystem, cohort: list[CohortEntry], config: ShapeWorksConfig
) -> None:
    """Raise if the particles look degenerate (scattered well beyond the organ); see config."""
    if config.max_spread_ratio is None:
        return
    ratio = particle_spread_ratio(system, cohort)
    if np.isfinite(ratio) and ratio > config.max_spread_ratio:
        raise ValueError(
            f"particle correspondence looks degenerate: median spread ratio {ratio:.2f} exceeds "
            f"max_spread_ratio={config.max_spread_ratio} (particles span ~{ratio:.1f}x the organ "
            "extent, i.e. they scattered off the surface -- usually a bad distance transform from "
            "grooming). Re-groom + re-optimize (delete the cached particles.npz first), or relax "
            "ShapeWorksConfig.max_spread_ratio if this cohort is genuinely this spread out."
        )


def _foreground_voxels(img: Any) -> int:
    """Number of foreground voxels (>0.5) in a binary ShapeWorks ``Image``."""
    return int(np.count_nonzero(img.toArray() > 0.5))


def _binary_seg_image(entry: CohortEntry, dest: Path, sw: object) -> Any:
    """Write the whole-gland mask as a NRRD and load it back as a binary ShapeWorks ``Image``."""
    if entry.organ.array is None:
        raise ValueError(f"case {entry.case_id!r}: organ pixels not loaded")
    mask = (entry.organ.array > 0).astype(np.uint8)
    src = entry.organ.sitk_image
    img = sitk.GetImageFromArray(mask)
    if src is not None:
        img.CopyInformation(src)
    seg_path = dest / f"{_sanitize(entry.case_id)}.nrrd"
    sitk.WriteImage(img, str(seg_path))
    return sw.Image(str(seg_path))  # type: ignore[attr-defined]


def _pregroom_binary(entry: CohortEntry, config: ShapeWorksConfig, dest: Path, sw: object) -> Any:
    """Grooming step 1 for one subject: antialias -> iso resample -> re-binarize -> COM align.

    Shared by :func:`_groom_cohort` and :mod:`shape_modeling.analysis.reference`, so the reference
    selection sees exactly the pre-alignment shapes the grooming uses.
    """
    iso, anti = config.iso_spacing, config.antialias_iterations
    img = _binary_seg_image(entry, dest, sw)
    img.antialias(anti).resample(iso, sw.InterpolationType.Linear).binarize()  # type: ignore[attr-defined]
    img.antialias(anti)
    img.translate(img.center() - img.centerOfMass()).binarize()  # COM -> image centre
    if _foreground_voxels(img) == 0:
        raise ValueError(
            f"case {entry.case_id!r}: groomed mask is EMPTY after antialias/resample/COM-align "
            f"(antialias_iterations={anti}, iso_spacing={iso}). The organ mask is likely so "
            "small that antialias + resample eroded it away -- check the QC volume for it."
        )
    return img


def _groom_cohort(
    cohort: list[CohortEntry], config: ShapeWorksConfig, dest: Path, sw: object
) -> list[Path]:
    """Groom the whole cohort to aligned signed distance transforms; return DT paths in order.

    Follows the ShapeWorks recommended segmentation workflow so the groomed shapes are
    pose-normalised (a prerequisite the optimiser's own Procrustes cannot supply -- that only
    aligns particles, not the groomed images shown in Studio):

      1. per subject: antialias -> isotropic resample -> RE-BINARIZE -> centre-of-mass align;
      2. pick a medoid reference and rigidly (ICP) align every shape to it;
      3. fit each aligned mask to its own bounding box + margin -> signed distance transform.

    Three non-obvious points:
      * RE-BINARIZE after resample: resampling anisotropic slices to 1 mm with linear interpolation
        SMEARS the 0-crossing into a thick band; binarize() re-thresholds it to a crisp 0/1 mask so
        computeDT(0.5) takes distance to the 0/1 midpoint -- the true surface. (computeDT(0.0) after
        binarize is degenerate: 0 is background, so "the surface" becomes the whole background.)
      * RESAMPLE THE ALIGNMENT ONTO THE REFERENCE'S GRID, not each subject's own grid: applying a
        rigid transform in place keeps the subject's original grid, so a rotation/translation that
        swings the gland past that grid clips it (no fixed pad reliably absorbs every subject's
        rotation -- case '054' overran a 20-voxel pad). Resampling onto the (padded) reference grid
        lands every aligned gland inside that frame by construction. See step 2.
      * fitRegion (not crop) in step 3: a gland cut off at the FOV edge (e.g. Prostate158 094 at
        z-slice 0) would sit on the boundary where the optimiser queries the DT just outside it;
        fitRegion pads with background so the DT keeps a margin on every face.

    ICP (``createRigidRegistrationTransform``) requires *distance transforms*, so the per-subject
    DTs are built first, used to choose + align to the reference, then the transform is applied to
    the binary mask (the final DT is recomputed after cropping). All subjects' images are held in
    memory at once (the cohort is ~10^2 cases); the per-image crop in step 3 keeps that bounded.
    """
    # 1. antialias -> isotropic resample -> re-binarize -> centre-of-mass align (per subject).
    binaries: list[Any] = [_pregroom_binary(entry, config, dest, sw) for entry in cohort]

    # 2. reference selection + rigid (ICP) alignment to it. ICP needs DTs; the transform it yields
    #    is applied to the binary mask, which is re-binarised to stay crisp.
    #
    #    The transform is resampled onto a COMMON output grid (the reference's grid, padded), NOT
    #    the subject's own grid. Applying a transform in place keeps the subject's original grid,
    #    so any rotation/translation that swings the gland past that grid clips it -- and a pad big
    #    enough to absorb every subject's rotation is hard to pick (case '054' overran a 20-voxel
    #    pad). Since every shape is aligned *to* the reference, resampling onto the reference's grid
    #    lands each gland where the reference gland sits, inside that frame by construction; padding
    #    the frame by pad_voxels leaves room for glands a little larger than the reference.
    dts = [b.copy().computeDT(0.5) for b in binaries]
    if config.reference_index is not None:
        ref_idx = int(config.reference_index)
        if not 0 <= ref_idx < len(cohort):
            raise ValueError(
                f"ShapeWorksConfig.reference_index={ref_idx} out of range for {len(cohort)} shapes"
            )
        print(f"grooming reference: index {ref_idx} ({cohort[ref_idx].case_id}), set explicitly")
    else:
        if len(cohort) > 30:
            print(
                f"WARNING: {len(cohort)} shapes > 30 -- ShapeWorks' find_reference_image_index "
                "does not return the medoid here (random-subset argmin bug); set "
                "ShapeWorksConfig.reference_index (see shape_modeling.analysis.reference)."
            )
        ref_idx = sw.find_reference_image_index(dts)  # type: ignore[attr-defined]
        print(f"grooming reference: index {ref_idx} ({cohort[ref_idx].case_id}), from ShapeWorks")
    ref = dts[ref_idx].copy()
    frame = binaries[ref_idx].copy().pad(config.pad_voxels, 0)  # common output grid for every shape
    out = (frame.origin(), frame.dims(), frame.spacing(), frame.coordsys())
    for entry, binary, dt in zip(cohort, binaries, dts, strict=True):
        transform = dt.createRigidRegistrationTransform(
            ref, config.align_iso_value, config.icp_iterations
        )
        binary.applyTransform(transform, *out, sw.InterpolationType.Linear).binarize()  # type: ignore[attr-defined]
        if _foreground_voxels(binary) == 0:
            raise ValueError(
                f"case {entry.case_id!r}: groomed mask is EMPTY after rigid (ICP) alignment to the "
                "reference -- the aligned gland fell outside the reference frame (padded by "
                f"{config.pad_voxels} voxels). This case aligns poorly to the reference; inspect, "
                "or increase ShapeWorksConfig.pad_voxels if its gland is just larger than the "
                "reference's."
            )
    del dts, ref, frame  # free the pre-alignment DTs / frame before recomputing the final ones

    # 3. crop each aligned mask to its own bounding box + margin -> DT -> smooth -> write.
    dt_paths: list[Path] = []
    for entry, binary in zip(cohort, binaries, strict=True):
        bbox = sw.ImageUtils.boundingBox([binary], 0.5).pad(config.pad_voxels)  # type: ignore[attr-defined]
        # fitRegion (not crop): when a gland sits within pad_voxels of an image face -- e.g. one
        # flush against the FOV edge, or pushed there by the rigid transform above -- the padded
        # bbox spills past the grid and crop() raises "Invalid region ... outside physical bounds".
        # fitRegion crops where the region is inside the grid and PADS with background (0) where it
        # is not, so the DT always gets a clean margin on every face. Identical to crop otherwise.
        binary.fitRegion(bbox).computeDT(0.5)  # distance to the 0/1 midpoint (the surface)
        if config.blur_sigma > 0:
            binary.gaussianBlur(config.blur_sigma)  # smoother surfaces + gentler gradients
        dt_path = dest / f"{_sanitize(entry.case_id)}_DT.nrrd"
        binary.write(str(dt_path))
        dt_paths.append(dt_path)
    return dt_paths


def _optimization_parameters(config: ShapeWorksConfig) -> dict[str, str]:
    """Optimizer parameters for a ShapeWorks project."""
    return {
        "number_of_particles": str(config.num_particles),
        "iterations_per_split": str(config.iterations),
        "optimization_iterations": str(config.iterations),
        "starting_regularization": str(config.starting_regularization),
        "ending_regularization": str(config.ending_regularization),
        "relative_weighting": str(config.relative_weighting),
        "narrow_band": str(config.narrow_band),
        "use_normals": "0",
        "procrustes": "1",
        "procrustes_interval": "10",
    }


def run_shapeworks(
    cohort: list[CohortEntry], config: ShapeWorksConfig | None = None
) -> ParticleSystem:
    """Groom the cohort, optimize particles, and return the corresponded :class:`ParticleSystem`.

    Caches the result in ``config.work_dir/particles.npz`` and reuses it when the cohort's
    case ids match, so re-running (e.g. from a notebook) skips the expensive optimize. Both the
    cached and the freshly-optimized particles are sanity-checked with :func:`particle_spread_ratio`
    (see ``ShapeWorksConfig.max_spread_ratio``), so a stale/degenerate cache is rejected rather than
    silently analysed. Requires ShapeWorks (see module docstring).
    """
    config = config or ShapeWorksConfig()
    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = work_dir / "particles.npz"
    case_ids = [e.case_id for e in cohort]

    if config.use_cache and cache.exists():
        cached = ParticleSystem.load(cache)
        if cached.case_ids == case_ids:
            _check_particles(cached, cohort, config)  # refuse a stale/degenerate cache
            return cached

    sw = _require_shapeworks()

    groom_dir = work_dir / "groomed"
    groom_dir.mkdir(exist_ok=True)
    dt_paths = _groom_cohort(cohort, config, groom_dir, sw)

    # Build a ShapeWorks project of the groomed shapes and run the optimizer via the CLI.
    project = sw.Project()  # type: ignore[attr-defined]
    subjects = []
    for cid, dt in zip(case_ids, dt_paths, strict=True):
        subject = sw.Subject()  # type: ignore[attr-defined]
        subject.set_display_name(_sanitize(cid))
        subject.set_number_of_domains(1)
        subject.set_groomed_filenames([str(dt)])
        subjects.append(subject)
    project.set_subjects(subjects)
    params = sw.Parameters()  # type: ignore[attr-defined]
    for key, value in _optimization_parameters(config).items():
        params.set(key, sw.Variant(value))  # type: ignore[attr-defined]
    project.set_parameters("optimize", params)

    project_path = work_dir / "shape_model.swproj"
    project.save(str(project_path))
    result = subprocess.run(
        ["shapeworks", "optimize", "--name", str(project_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # ShapeWorks reports the real cause (bad DT, padding, etc.) on stdout/stderr;
        # surface it instead of a bare exit code.
        raise RuntimeError(
            f"`shapeworks optimize` failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )

    # The optimizer names particle files after the groomed file stem, not the display name:
    # the groomed DTs are <sanitized_id>_DT.nrrd, so outputs are <sanitized_id>_DT_world.particles.
    particle_dir = next(
        (p.parent for p in work_dir.rglob("*_world.particles")),
        work_dir,
    )
    system = load_particles(particle_dir, case_ids, suffix="_DT_world.particles")
    _check_particles(system, cohort, config)  # catch a bad optimize before caching it
    system.save(cache)
    return system
