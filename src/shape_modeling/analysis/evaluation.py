"""SSM quality metrics -- compactness, generalization, specificity (Davies et al. 2002).

These are the three standard curves for judging a Statistical Shape Model, each plotted
**against the number of retained PCA modes** ``M``. They answer complementary questions:

* **compactness(M)** -- the cumulative fraction of total variance the first ``M`` modes
  capture. *Higher is better*: a compact model describes the cohort with few modes.
* **generalization(M)** -- leave-one-out reconstruction error in **mm**. Build PCA on the
  other ``S-1`` shapes, project the held-out shape onto ``M`` modes and measure how far the
  reconstruction lands from the truth (mean per-particle Euclidean distance). *Lower is
  better*: it shows the model can represent shapes it was **not** trained on. The curve
  falls then flattens; a rise at high ``M`` is overfitting to noise.
* **specificity(M)** -- draw random shapes from the model (mode weights
  ``b ~ N(0, eigenvalue)``) and measure the mean per-particle distance (mm) to the nearest
  **training** shape. *Lower is better*: it shows the model only generates **valid**
  anatomy. The curve rises with ``M`` as extra modes let samples wander off the data.

The implementation operates on a corresponded particle stack ``(S, M, 3)`` -- e.g. the
ShapeWorks ``*_world.particles`` in millimetres held by
:class:`~shape_modeling.analysis.correspondence.ParticleSystem` -- and reproduces ShapeWorks
Studio's own ``compactness.csv`` / ``generalization.csv`` **exactly** on the same particles
(see ``tests/test_analysis.py``). Specificity uses random sampling, so its absolute level
depends on ``n_samples`` and ``seed`` and will not bit-match Studio -- read the monotone
trend, not the third decimal.

Pure numpy + scikit-learn; no ShapeWorks dependency, so every notebook can evaluate its own
correspondence. :func:`load_shapeworks_evaluation` reads Studio's CSVs back into the same
:class:`SSMEvaluation` container for an apples-to-apples overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "SSMEvaluation",
    "compactness",
    "default_mode_grid",
    "evaluate_ssm",
    "generalization",
    "load_shapeworks_evaluation",
    "specificity",
]


@dataclass
class SSMEvaluation:
    """The three SSM quality curves sampled at a shared set of mode counts.

    ``modes`` is the x-axis (number of retained PCA modes); the three metric arrays line up
    with it element-for-element. ``n_subjects`` / ``n_points`` record the cohort the curves
    were measured on. ``label`` is a free-text tag for plot legends / overlays.
    """

    modes: np.ndarray  # (G,) mode counts, ascending
    compactness: np.ndarray  # (G,) cumulative explained-variance ratio, in [0, 1]
    generalization: np.ndarray  # (G,) leave-one-out reconstruction error (mm)
    specificity: np.ndarray  # (G,) nearest-training-neighbour distance of model samples (mm)
    n_subjects: int = 0
    n_points: int = 0
    label: str = ""

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy :class:`pandas.DataFrame` (one row per mode count)."""
        import pandas as pd

        return pd.DataFrame(
            {
                "modes": self.modes,
                "compactness": self.compactness,
                "generalization_mm": self.generalization,
                "specificity_mm": self.specificity,
            }
        )

    def modes_for_variance(self, fraction: float) -> int:
        """Smallest mode count whose compactness reaches ``fraction`` (e.g. 0.90)."""
        hit = int(np.searchsorted(self.compactness, fraction))
        return int(self.modes[min(hit, len(self.modes) - 1)])


def _flatten(particles: np.ndarray) -> tuple[np.ndarray, int]:
    """Validate an ``(S, M, 3)`` stack and return ``((S, 3M) matrix, n_points)``."""
    pts = np.asarray(particles, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise ValueError(f"expected (S, M, 3), got {pts.shape}")
    return pts.reshape(pts.shape[0], -1), pts.shape[1]


def default_mode_grid(max_modes: int) -> np.ndarray:
    """Mode counts to sample the curves at: dense for the first 15, then thinned.

    Computing specificity at *every* mode is expensive for large cohorts (each mode count
    re-samples the model), so for big models we sample a representative grid -- the curves
    are smooth, so this is visually indistinguishable from the full sweep while keeping the
    whole evaluation to ~a minute.
    """
    if max_modes <= 0:
        return np.array([], dtype=int)
    if max_modes <= 20:
        return np.arange(1, max_modes + 1)
    grid = list(range(1, 16))  # 1..15 dense -- where the curves bend most
    grid += list(range(20, min(max_modes, 50) + 1, 5))  # 20, 25, ..., 50
    if max_modes > 50:
        grid += np.unique(np.round(np.geomspace(60, max_modes, 8)).astype(int)).tolist()
    grid.append(max_modes)
    return np.array(sorted({m for m in grid if 1 <= m <= max_modes}), dtype=int)


def compactness(particles: np.ndarray, modes: np.ndarray | None = None) -> np.ndarray:
    """Cumulative explained-variance ratio of the first ``M`` PCA modes, at each ``M``.

    Scale-free (depends only on eigenvalue ratios), so it is identical whether the input is
    in mm or unit-normalised.
    """
    flat, _ = _flatten(particles)
    n = flat.shape[0]
    centred = flat - flat.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    eig = sv**2 / (n - 1)
    cum = np.cumsum(eig) / eig.sum()
    if modes is None:
        return cum
    return cum[np.asarray(modes, dtype=int) - 1]


def generalization(particles: np.ndarray, modes: np.ndarray | None = None) -> np.ndarray:
    """Leave-one-out reconstruction error (mean per-particle Euclidean distance, mm).

    For each subject: PCA the other ``S-1`` shapes, project the held-out shape onto the first
    ``M`` modes, and measure the mean distance between its particles and the reconstruction;
    average over subjects. Matches ShapeWorks Studio's ``generalization.csv`` exactly.
    """
    flat, n_points = _flatten(particles)
    n = flat.shape[0]
    max_modes = min(n - 2, flat.shape[1])  # LOO trains on n-1 shapes -> n-2 usable modes
    if max_modes < 1:
        raise ValueError("need at least 3 subjects to measure generalization")

    err = np.zeros(max_modes)
    for leave in range(n):
        train = np.delete(flat, leave, axis=0)
        mu = train.mean(axis=0)
        _, _, vt = np.linalg.svd(train - mu, full_matrices=False)
        basis = vt[:max_modes]  # (max_modes, 3M)
        residual = flat[leave] - mu
        coeff = basis @ residual  # (max_modes,)
        # Recon offset after keeping the first m modes = cumulative sum of coeff_k * basis_k;
        # residual-to-truth after m modes is then `residual - that`. Build every m at once.
        partial = np.cumsum(coeff[:, None] * basis, axis=0)  # (max_modes, 3M)
        resid_m = residual[None, :] - partial  # (max_modes, 3M)
        per_particle = np.linalg.norm(resid_m.reshape(max_modes, n_points, 3), axis=2)
        err += per_particle.mean(axis=1)
    err /= n

    if modes is None:
        return err
    idx = np.clip(np.asarray(modes, dtype=int) - 1, 0, max_modes - 1)
    return err[idx]


def specificity(
    particles: np.ndarray,
    modes: np.ndarray,
    *,
    n_samples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Mean distance (mm) from random model samples to their nearest training shape.

    For each mode count in ``modes``: draw ``n_samples`` shapes with weights
    ``b ~ N(0, eigenvalue)`` along the first ``M`` modes, and average, over samples, the mean
    per-particle distance to the closest training shape. Stochastic -- fix ``seed`` for
    reproducibility; absolute values depend on ``n_samples`` and will not match Studio's CSV
    to the digit, but the rising trend reproduces.
    """
    flat, n_points = _flatten(particles)
    n = flat.shape[0]
    train = flat.reshape(n, n_points, 3)
    mu = flat.mean(axis=0)
    _, sv, vt = np.linalg.svd(flat - mu, full_matrices=False)
    eig = sv**2 / (n - 1)
    rng = np.random.default_rng(seed)

    modes = np.asarray(modes, dtype=int)
    out = np.zeros(len(modes))
    batch = 100
    for j, m in enumerate(modes):
        m = int(min(m, len(eig)))
        basis = vt[:m]
        sd = np.sqrt(eig[:m])
        total = 0.0
        for start in range(0, n_samples, batch):
            size = min(batch, n_samples - start)
            weights = rng.standard_normal((size, m)) * sd
            samples = (mu + weights @ basis).reshape(size, 1, n_points, 3)
            dist = np.linalg.norm(samples - train[None], axis=3).mean(axis=2)  # (size, n)
            total += dist.min(axis=1).sum()
        out[j] = total / n_samples
    return out


def evaluate_ssm(
    particles: np.ndarray,
    *,
    modes: np.ndarray | None = None,
    n_specificity_samples: int = 1000,
    seed: int = 0,
    label: str = "",
) -> SSMEvaluation:
    """Compute all three SSM quality curves on a ``(S, M, 3)`` corresponded particle stack.

    Pass the ShapeWorks world particles (in mm) -- ``system.particles`` -- so generalization
    and specificity come out in millimetres. ``modes`` defaults to :func:`default_mode_grid`;
    pass an explicit array to control the x-axis.
    """
    flat, _ = _flatten(particles)
    n = flat.shape[0]
    max_modes = min(n - 2, flat.shape[1])
    grid = default_mode_grid(max_modes) if modes is None else np.asarray(modes, dtype=int)

    return SSMEvaluation(
        modes=grid,
        compactness=compactness(particles, grid),
        generalization=generalization(particles, grid),
        specificity=specificity(particles, grid, n_samples=n_specificity_samples, seed=seed),
        n_subjects=n,
        n_points=int(flat.shape[1] // 3),
        label=label,
    )


def load_shapeworks_evaluation(
    work_dir: str | Path, *, label: str = "ShapeWorks Studio"
) -> SSMEvaluation:
    """Read Studio's ``compactness.csv`` / ``generalization.csv`` / ``specificity.csv``.

    Each is ``# Number of Modes, <metric>``. Returns them in the same :class:`SSMEvaluation`
    container as :func:`evaluate_ssm` for a direct overlay. Missing files become NaN columns.
    """
    work_dir = Path(work_dir)

    def _read(name: str) -> tuple[np.ndarray, np.ndarray]:
        path = work_dir / f"{name}.csv"
        if not path.exists():
            return np.array([]), np.array([])
        data = np.loadtxt(path, delimiter=",", comments="#")
        data = np.atleast_2d(data)
        return data[:, 0].astype(int), data[:, 1]

    modes, comp = _read("compactness")
    g_modes, gen = _read("generalization")
    s_modes, spec = _read("specificity")
    if modes.size == 0:
        modes = g_modes if g_modes.size else s_modes

    def _align(src_modes: np.ndarray, values: np.ndarray) -> np.ndarray:
        if src_modes.size == 0:
            return np.full(modes.shape, np.nan)
        lookup = dict(zip(src_modes.tolist(), values.tolist(), strict=False))
        return np.array([lookup.get(int(m), np.nan) for m in modes])

    return SSMEvaluation(
        modes=modes,
        compactness=_align(modes, comp) if comp.size else np.full(modes.shape, np.nan),
        generalization=_align(g_modes, gen),
        specificity=_align(s_modes, spec),
        label=label,
    )
