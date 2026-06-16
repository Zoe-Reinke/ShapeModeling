"""Stage 7 -- serialise a :class:`ShapeModel` for downstream parameter fitting.

The SSM built by :mod:`~shape_modeling.analysis.ssm` is, at heart, three arrays: a mean
shape, a set of principal modes, and their variances. :func:`export_model` writes them
(plus optional mesh connectivity and provenance metadata) to a single ``.npz`` so a
*separate* program -- e.g. an ultrasound fitting / optimisation loop -- can reload the model
with :func:`load_model` and reconstruct shapes from a handful of parameters without depending
on this package, scikit-learn, or the original cohort.

Parameterisation. A shape is

    x(b) = mean + sum_k  b_k * sqrt(eigenvalue_k) * mode_k

where the weights ``b`` are **dimensionless, in units of standard deviation**: ``b_k = 0`` is
the mean, ``|b_k| <= 3`` spans the plausible range of the cohort. These are the knobs an
optimiser turns. :meth:`ExportedModel.instance` evaluates this; :meth:`ExportedModel.clamp`
enforces the plausibility box.

Pose at fit time. The model is built on :func:`~shape_modeling.analysis.ssm.gpa`-aligned
shapes with ``scale=True``, so the mean is **pose-free and unitless** (unit Frobenius norm,
~0.0x coordinates, *not* mm). To match it to a real anatomy in millimetre world coordinates
you therefore solve a 7-DOF similarity (uniform scale, rotation, translation) *alongside* the
shape weights. :meth:`ExportedModel.fit` does exactly this -- alternating a pose solve and a
shape-weight solve -- and :meth:`ExportedModel.instance` applies a ``(scale, rotation,
translation)`` pose when given one. The unitless model is the deliberate choice (it decouples
size from shape; see CLAUDE.md): downstream code owns the scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from shape_modeling.analysis.ssm import ShapeModel
from shape_modeling.analysis.surface import Surface, _similarity_fit

__all__ = [
    "ExportedModel",
    "FitResult",
    "Pose",
    "SurfaceFitResult",
    "export_model",
    "load_model",
    "surface_distance",
]

FORMAT_VERSION = 1

# A similarity pose mapping model coordinates into the target frame: ``s * (x @ R.T) + t``.
Pose = tuple[float, np.ndarray, np.ndarray]


@dataclass
class FitResult:
    """Outcome of fitting an :class:`ExportedModel` to a target point set."""

    weights: np.ndarray  # (K,) shape weights in SD units (b_k = 0 is the mean)
    scale: float  # uniform pose scale (1.0 if pose was disabled)
    rotation: np.ndarray  # (3, 3) proper rotation
    translation: np.ndarray  # (3,) translation in target units
    points: np.ndarray  # (M, 3) the fitted shape, posed into the target frame
    rmse: float  # RMS per-point distance to the target

    @property
    def pose(self) -> Pose:
        """The pose as the ``(scale, rotation, translation)`` triple ``instance`` expects."""
        return (self.scale, self.rotation, self.translation)


@dataclass
class SurfaceFitResult:
    """Outcome of fitting an :class:`ExportedModel` to a *non-corresponded* target surface.

    Unlike :class:`FitResult`, there is no per-point target to take an RMSE against (the
    correspondence is re-estimated each iteration), so model quality is reported as the
    **symmetric surface distance** (mm) between the fitted shape and the target cloud.
    """

    weights: np.ndarray  # (K,) shape weights in SD units (b_k = 0 is the mean)
    scale: float  # uniform pose scale
    rotation: np.ndarray  # (3, 3) proper rotation
    translation: np.ndarray  # (3,) translation in target units
    points: np.ndarray  # (M, 3) the fitted shape, posed into the target frame
    mean_distance: float  # symmetric mean surface distance to target (mm) -- the headline metric
    rms_distance: float  # symmetric RMS surface distance (mm)
    hausdorff95: float  # symmetric 95th-percentile surface distance (mm)
    n_iter: int  # ICP iterations the winning restart actually ran
    converged: bool  # whether that restart hit the distance-change tolerance before n_iter

    @property
    def pose(self) -> Pose:
        """The pose as the ``(scale, rotation, translation)`` triple ``instance`` expects."""
        return (self.scale, self.rotation, self.translation)


@dataclass
class ExportedModel:
    """A self-contained, reloadable shape model: mean, modes, variances, and connectivity.

    Mirrors the arrays of :class:`~shape_modeling.analysis.ssm.ShapeModel` but adds the
    reconstruction/fitting helpers a downstream consumer needs, and is detached from
    scikit-learn so it round-trips through a plain ``.npz``.
    """

    mean: np.ndarray  # (M, 3)
    components: np.ndarray  # (K, 3M) unit principal modes
    eigenvalues: np.ndarray  # (K,) variance along each mode
    explained_variance_ratio: np.ndarray  # (K,)
    faces: np.ndarray | None = None  # (F, 3) shared connectivity, or None
    metadata: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return self.mean.shape[0]

    @property
    def n_modes(self) -> int:
        return self.components.shape[0]

    def instance(
        self, weights: np.ndarray | None = None, *, pose: Pose | None = None
    ) -> np.ndarray:
        """Reconstruct a shape from mode ``weights`` (SD units), optionally posed.

        ``weights`` may be shorter than :attr:`n_modes` -- the remaining modes are held at the
        mean. ``weights=None`` returns the mean itself. When ``pose`` is a ``(scale, rotation,
        translation)`` triple the result is mapped into the target frame as
        ``scale * (x @ rotation.T) + translation``. Returns an ``(M, 3)`` point set.
        """
        x = self.mean.copy()
        if weights is not None:
            b = np.asarray(weights, dtype=np.float64).ravel()
            if b.size > self.n_modes:
                raise ValueError(f"got {b.size} weights but only {self.n_modes} modes")
            k = b.size
            scores = b * np.sqrt(self.eigenvalues[:k])  # SD units -> coordinate units
            x = x + (scores @ self.components[:k]).reshape(self.n_points, 3)
        if pose is not None:
            scale, rot, translation = pose
            x = scale * (x @ np.asarray(rot).T) + np.asarray(translation)
        return x

    def clamp(self, weights: np.ndarray, *, n_sd: float = 3.0) -> np.ndarray:
        """Clip mode ``weights`` to the plausible ``+/- n_sd`` box (a new array)."""
        return np.clip(np.asarray(weights, dtype=np.float64), -n_sd, n_sd)

    def surface(self, weights: np.ndarray | None = None, *, pose: Pose | None = None) -> Surface:
        """A renderable :class:`Surface` for ``weights``; requires exported ``faces``."""
        if self.faces is None:
            raise ValueError("no faces stored in this model; export with faces=... for surface()")
        return Surface(verts=self.instance(weights, pose=pose), faces=self.faces)

    def fit(
        self,
        target: np.ndarray,
        *,
        n_modes: int | None = None,
        pose: bool = True,
        n_iter: int = 10,
        max_sd: float = 3.0,
    ) -> FitResult:
        """Fit shape weights (and, by default, a similarity pose) to a target point set.

        ``target`` is an ``(M, 3)`` point set **in correspondence with the model points** (same
        count, same ordering as :attr:`mean`) -- e.g. landmarks or a corresponded surface
        sampled from the real anatomy. The fit alternates two closed-form solves:

        1. **Pose** -- the least-squares similarity carrying the current shape onto ``target``
           (Umeyama; reused from :func:`~shape_modeling.analysis.surface._similarity_fit`).
        2. **Shape** -- project ``target`` (pulled back into the model frame) onto the modes
           and rescale to SD units, then :meth:`clamp` to ``+/- max_sd`` so the result stays a
           plausible member of the cohort.

        ``pose=False`` skips step 1 (use when ``target`` is already in the model's unitless
        frame). ``n_modes`` limits how many modes the fit may move (defaults to all). The
        returned :class:`FitResult` carries the weights, the pose, the posed points, and RMSE.
        """
        tgt = np.asarray(target, dtype=np.float64)
        if tgt.shape != self.mean.shape:
            raise ValueError(
                f"target must be in correspondence with the model: expected {self.mean.shape}, "
                f"got {tgt.shape}"
            )
        k = self.n_modes if n_modes is None else min(n_modes, self.n_modes)
        sqrt_lambda = np.sqrt(self.eigenvalues[:k])
        comps = self.components[:k]  # (k, 3M), unit vectors

        b = np.zeros(k)
        scale, rot, translation = 1.0, np.eye(3), np.zeros(3)
        for _ in range(n_iter):
            x = self.instance(b)  # current shape, model frame
            if pose:
                scale, rot, translation = _similarity_fit(x, tgt)  # x -> target
                # Pull the target back into the model frame: invert s*(x@R.T)+t.
                tgt_model = ((tgt - translation) / scale) @ rot
            else:
                tgt_model = tgt
            resid = (tgt_model - self.mean).reshape(-1)  # (3M,)
            scores = comps @ resid  # projection onto each unit mode
            b = self.clamp(scores / sqrt_lambda, n_sd=max_sd)

        posed = self.instance(b, pose=(scale, rot, translation) if pose else None)
        rmse = float(np.sqrt(np.mean(np.sum((posed - tgt) ** 2, axis=1))))
        return FitResult(
            weights=b,
            scale=float(scale),
            rotation=np.asarray(rot, dtype=np.float64),
            translation=np.asarray(translation, dtype=np.float64),
            points=posed,
            rmse=rmse,
        )

    def fit_surface(
        self,
        target: np.ndarray | Surface,
        *,
        n_modes: int | None = None,
        n_iter: int = 50,
        tol: float = 1e-4,
        max_sd: float = 3.0,
        n_restarts: int = 4,
        init_pose: Pose | None = None,
    ) -> SurfaceFitResult:
        """Fit shape weights + a similarity pose to a **non-corresponded** target surface.

        This is the held-out validation path: it answers "how well can the model reconstruct an
        organ it never saw, given only an arbitrary mask of it?". Unlike :meth:`fit`, ``target``
        need not be in correspondence with the model -- it is an ``(N, 3)`` point cloud sampling
        the real surface (e.g. ``extract_surface(clean_mask(mask), geom).verts``), with ``N``
        arbitrary and in any order or pose. A :class:`~shape_modeling.analysis.surface.Surface`
        is accepted directly (its ``verts`` are used).

        The fit is **iterated closest point** with a shape solve in the loop. Each iteration:

        1. pose the current model instance into the target frame;
        2. match each model point to its nearest target point -- a *pseudo*-correspondence,
           re-estimated every iteration (this is what removes the need for true correspondence);
        3. re-solve the similarity pose (Umeyama) and the mode weights (project onto the modes,
           rescale to SD units, :meth:`clamp` to ``+/- max_sd``), exactly as :meth:`fit` does on
           the pseudo-corresponded target.

        ICP only finds a local optimum, so the fit is run from ``n_restarts`` initial poses -- the
        model's principal axes flipped to each proper-rotation sign combination -- and the
        lowest-distance result is kept. Pass ``init_pose`` to use a single known initialisation
        instead (e.g. when ``target`` is already roughly aligned after grooming), skipping the
        search. The returned :class:`SurfaceFitResult` carries the weights, pose, fitted points,
        and the symmetric surface distance to the target -- the number to report for validation.

        Clean the mask and extract its surface before calling; this method does no smoothing.
        """
        verts = target.verts if isinstance(target, Surface) else target
        tgt = np.asarray(verts, dtype=np.float64)
        if tgt.ndim != 2 or tgt.shape[1] != 3:
            raise ValueError(f"target must be an (N, 3) point cloud, got {tgt.shape}")

        k = self.n_modes if n_modes is None else min(n_modes, self.n_modes)
        sqrt_lambda = np.sqrt(self.eigenvalues[:k])
        comps = self.components[:k]  # (k, 3M)
        tree = cKDTree(tgt)

        inits = (
            [init_pose]
            if init_pose is not None
            else _init_surface_poses(self.mean, tgt, n_restarts)
        )

        best: SurfaceFitResult | None = None
        for scale, rot, translation in inits:
            b = np.zeros(k)
            prev = np.inf
            converged = False
            used = 0
            for _ in range(n_iter):
                used += 1
                x = self.instance(b)  # current shape, model frame
                posed = scale * (x @ np.asarray(rot).T) + np.asarray(translation)
                dist, idx = tree.query(posed)
                matched = tgt[idx]  # (M, 3) pseudo-correspondence in the target frame
                scale, rot, translation = _similarity_fit(x, matched)  # x -> matched
                matched_model = ((matched - translation) / scale) @ rot  # pull back to model frame
                resid = (matched_model - self.mean).reshape(-1)
                b = self.clamp((comps @ resid) / sqrt_lambda, n_sd=max_sd)
                mean_d = float(dist.mean())
                if abs(prev - mean_d) < tol:
                    converged = True
                    break
                prev = mean_d

            posed = self.instance(b, pose=(scale, rot, translation))
            mean_d, rms_d, hd95 = surface_distance(posed, tgt)
            if best is None or mean_d < best.mean_distance:
                best = SurfaceFitResult(
                    weights=b,
                    scale=float(scale),
                    rotation=np.asarray(rot, dtype=np.float64),
                    translation=np.asarray(translation, dtype=np.float64),
                    points=posed,
                    mean_distance=mean_d,
                    rms_distance=rms_d,
                    hausdorff95=hd95,
                    n_iter=used,
                    converged=converged,
                )
        assert best is not None  # `inits` is always non-empty
        return best


def export_model(
    model: ShapeModel,
    path: str | Path,
    *,
    faces: np.ndarray | None = None,
    units: str = "normalized",
    metadata: dict | None = None,
) -> Path:
    """Write ``model`` to a single ``.npz`` for downstream reload with :func:`load_model`.

    Stores the mean, principal modes, eigenvalues, and explained-variance ratio, plus optional
    mesh ``faces`` (shared connectivity so vertex ``j`` is comparable across reconstructions --
    e.g. ``hull_surface(model.mean).faces`` or a :func:`warp_surface` template's faces) and a
    free-form ``metadata`` dict (provenance: cohort, harmonization, date).

    ``units`` records the coordinate frame the model lives in -- ``"normalized"`` for a
    ``gpa(scale=True)`` model (unitless ~0.0x coords; uniform scale must be recovered at fit
    time) or ``"mm"`` for a ``gpa(scale=False)`` model (already millimetres; only rotation +
    translation are solved at fit time). Pose is solved at fit time either way. The metadata is
    embedded in the archive as JSON, so the ``.npz`` is the single source of truth.

    Returns the written path (``.npz`` suffix ensured).
    """
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "format_version": FORMAT_VERSION,
        "n_points": int(model.n_points),
        "n_modes": int(model.n_modes),
        "units": units,
        "pose_at_fit_time": True,
        "parameterization": "x = mean + sum_k b_k * sqrt(eigenvalue_k) * mode_k  (b in SD units)",
        **(metadata or {}),
    }

    arrays: dict[str, np.ndarray] = {
        "mean": np.asarray(model.mean, dtype=np.float64),
        "components": np.asarray(model.components, dtype=np.float64),
        "eigenvalues": np.asarray(model.eigenvalues, dtype=np.float64),
        "explained_variance_ratio": np.asarray(model.explained_variance_ratio, dtype=np.float64),
        "metadata_json": np.array(json.dumps(meta)),
    }
    if faces is not None:
        arrays["faces"] = np.asarray(faces, dtype=np.int64)

    np.savez(path, **arrays)  # type: ignore[arg-type]
    return path


def load_model(path: str | Path) -> ExportedModel:
    """Load an :class:`ExportedModel` written by :func:`export_model`.

    Self-contained: needs only numpy, not this package's analysis dependencies.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        version = meta.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported format_version {version!r} (expected {FORMAT_VERSION})")
        return ExportedModel(
            mean=data["mean"],
            components=data["components"],
            eigenvalues=data["eigenvalues"],
            explained_variance_ratio=data["explained_variance_ratio"],
            faces=data["faces"] if "faces" in data.files else None,
            metadata=meta,
        )


def surface_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Symmetric surface distance between two point clouds (mm): ``(mean, rms, hausdorff95)``.

    Nearest-neighbour distance is measured in both directions (every point of ``a`` to its closest
    point in ``b`` and vice-versa) and the two sets pooled, so neither cloud's extent is
    privileged. Returns the pooled mean (an average-symmetric-surface-distance estimate), the RMS,
    and the 95th percentile (a robust Hausdorff). ``a`` and ``b`` need not have the same length.

    This is a point-set approximation: with dense surface samples (e.g. marching-cubes vertices, or
    a model with many particles) it tracks the true mesh ASSD closely; on a sparse particle set the
    closest *vertex* slightly overestimates the closest *surface* distance.
    """
    pa = np.asarray(a, dtype=np.float64)
    pb = np.asarray(b, dtype=np.float64)
    d_ab, _ = cKDTree(pb).query(pa)
    d_ba, _ = cKDTree(pa).query(pb)
    pooled = np.concatenate([d_ab, d_ba])
    return (
        float(pooled.mean()),
        float(np.sqrt(np.mean(pooled**2))),
        float(np.percentile(pooled, 95)),
    )


# All proper- and improper-rotation sign flips of a principal frame; filtered to det=+1 per call.
_SIGN_FLIPS = np.array(
    [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    ],
    dtype=np.float64,
)


def _principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centroid, principal axes (columns, variance-descending), and centroid size of a cloud."""
    mu = points.mean(axis=0)
    centred = points - mu
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    size = float(np.sqrt((centred**2).sum()))
    return mu, vt.T, size


def _init_surface_poses(mean: np.ndarray, target: np.ndarray, n_restarts: int) -> list[Pose]:
    """Initial similarity poses aligning the model's principal frame onto the target's.

    The two principal frames match only up to an axis-sign ambiguity, so one pose is emitted per
    proper-rotation (det=+1) sign flip -- up to ``n_restarts`` of them. ICP then refines each and
    the caller keeps the best, making :meth:`ExportedModel.fit_surface` robust to an arbitrarily
    oriented target without a manual initial guess.
    """
    mu_m, axes_m, size_m = _principal_axes(np.asarray(mean, dtype=np.float64))
    mu_t, axes_t, size_t = _principal_axes(np.asarray(target, dtype=np.float64))
    scale = size_t / size_m if size_m > 0 else 1.0

    # Keep only sign flips that make R = (flipped target axes) @ model axes^T a proper rotation.
    need = float(np.sign(np.linalg.det(axes_t) * np.linalg.det(axes_m))) or 1.0
    flips = [s for s in _SIGN_FLIPS if float(np.prod(s)) == need][: max(1, n_restarts)]

    poses: list[Pose] = []
    for s in flips:
        rot = (axes_t * s) @ axes_m.T  # maps each model axis onto the (sign-flipped) target axis
        translation = mu_t - scale * (rot @ mu_m)
        poses.append((float(scale), rot, translation))
    return poses
