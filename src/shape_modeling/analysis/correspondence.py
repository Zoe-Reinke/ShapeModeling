"""Particle-based correspondence across subjects via ShapeWorks.

ShapeWorks places ``M`` corresponding particles on every shape so that particle ``j`` marks
the same anatomical location on each subject -- the correspondence PCA needs. The flow is:
groom each whole-gland segmentation (antialias -> isotropic resample -> signed distance
transform), run the particle optimizer over the groomed shapes, then read the per-subject
``*_world.particles`` into an ``(S, M, 3)`` array.

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
from typing import TYPE_CHECKING

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
    pad_voxels: int = 20  # background margin (voxels) added to the 1 mm mask before the DT, so a
    # gland touching the scan's field-of-view edge still has room for the optimizer to query around
    blur_sigma: float = 1.0  # Gaussian smoothing (mm) of the DT; gentler gradients for optimizer
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


def _groom(entry: CohortEntry, config: ShapeWorksConfig, dest: Path, sw: object) -> Path:
    """Groom one whole-gland segmentation to a signed distance transform; return its path."""
    if entry.organ.array is None:
        raise ValueError(f"case {entry.case_id!r}: organ pixels not loaded")
    mask = (entry.organ.array > 0).astype(np.uint8)
    src = entry.organ.sitk_image
    img = sitk.GetImageFromArray(mask)
    if src is not None:
        img.CopyInformation(src)
    seg_path = dest / f"{_sanitize(entry.case_id)}.nrrd"
    sitk.WriteImage(img, str(seg_path))

    # antialias -> resample -> RE-BINARIZE -> PAD -> signed distance transform (ShapeWorks Image).
    # Three non-obvious steps, each fixing a way the DT goes degenerate (which scatters particles or
    # aborts the optimizer):
    #   * antialias makes a smooth signed field, but resampling anisotropic slices to 1 mm with
    #     linear interpolation SMEARS the 0-crossing into a thick band (~16% of voxels near 0).
    #     binarize() re-thresholds it back to a crisp 0/1 mask, then computeDT(0.5) takes distance
    #     to the 0/1 midpoint -- the true surface. (computeDT(0.0) after binarize is degenerate: 0
    #     is the background value, so "the surface" becomes the whole background -> inf distances.)
    #   * pad AFTER resample (not before): resample re-grids to the original physical extent and
    #     discards earlier padding, so padding first leaves the DT flush against the field of view.
    #     A gland cut off at the FOV edge (e.g. Prostate158 094 touches z-slice 0) then sits on the
    #     image boundary and the optimizer queries the DT just outside it ("...outside the given
    #     image domain"). Padding the 1 mm binary with background 0 gives every face a real margin.
    groomed = sw.Image(str(seg_path))  # type: ignore[attr-defined]
    groomed.antialias(config.antialias_iterations)
    groomed.resample([config.iso_spacing] * 3, sw.InterpolationType.Linear)  # type: ignore[attr-defined]
    groomed.binarize()  # collapse the resample-smeared interface back to a crisp 0/1 mask
    groomed.pad(config.pad_voxels, 0)  # 0 = background; AFTER resample so the margin survives
    groomed.computeDT(0.5)  # distance to the 0/1 midpoint (the surface)
    if config.blur_sigma > 0:
        groomed.gaussianBlur(config.blur_sigma)  # smooth the DT for gentler optimizer gradients
    dt_path = dest / f"{_sanitize(entry.case_id)}_DT.nrrd"
    groomed.write(str(dt_path))
    return dt_path


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
    dt_paths = [_groom(entry, config, groom_dir, sw) for entry in cohort]

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
