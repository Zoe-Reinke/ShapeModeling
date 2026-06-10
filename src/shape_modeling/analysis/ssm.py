"""Generalized Procrustes alignment and PCA -- the Statistical Shape Model core.

Given corresponded point sets (one per subject, same number of points in the same order --
e.g. ShapeWorks particles), :func:`gpa` removes pose (translation, rotation, and optionally
scale) so that only *shape* variation remains, and :func:`build_ssm` runs PCA over the
aligned points to give the mean shape, principal modes, eigenvalues, and per-subject scores.

The result is intentionally small and inspectable: scatter the scores, look at the
eigenvalue decay, and walk the mean along a mode with :func:`mode_shape`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

__all__ = [
    "ShapeModel",
    "build_ssm",
    "centroid_size",
    "gpa",
    "mode_shape",
]


def centroid_size(shape: np.ndarray) -> np.ndarray:
    """Procrustes centroid size: RMS-style distance of points from their centroid.

    Accepts a single ``(M, 3)`` shape (returns a scalar) or a stack ``(S, M, 3)`` (returns
    ``(S,)``). This is the standard scalar measure of overall shape *size*; correlate it
    against a mode's scores to test whether that mode is a size mode.

    Note: :func:`gpa` with ``scale=True`` normalises every shape to unit Frobenius norm, so
    the centroid size of *aligned* shapes is ~constant. To test mode 1 against size, compute
    this on the **raw** particles (before :func:`gpa`).
    """
    pts = np.asarray(shape, dtype=np.float64)
    if pts.ndim not in (2, 3) or pts.shape[-1] != 3:
        raise ValueError(f"expected (M, 3) or (S, M, 3), got {pts.shape}")
    centroid = pts.mean(axis=-2, keepdims=True)
    return np.sqrt(((pts - centroid) ** 2).sum(axis=(-2, -1)))


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimal rotation (proper, det=+1) aligning centred points ``a`` onto ``b``."""
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    return u @ correction @ vt


def gpa(
    particles: np.ndarray,
    *,
    scale: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Procrustes alignment of ``(S, M, 3)`` corresponded point sets.

    Removes per-shape translation and rotation (and scale when ``scale=True``) by iteratively
    aligning every shape to the evolving mean.

    Returns:
        ``(aligned, mean)`` where ``aligned`` is ``(S, M, 3)`` and ``mean`` is ``(M, 3)``.
    """
    shapes = np.asarray(particles, dtype=np.float64)
    if shapes.ndim != 3 or shapes.shape[2] != 3:
        raise ValueError(f"expected (S, M, 3), got {shapes.shape}")

    # Centre (and optionally scale to unit Frobenius norm) every shape.
    shapes = shapes - shapes.mean(axis=1, keepdims=True)
    if scale:
        norms = np.linalg.norm(shapes, axis=(1, 2), keepdims=True)
        shapes = shapes / np.where(norms == 0, 1.0, norms)

    mean = shapes[0].copy()
    for _ in range(max_iter):
        for i in range(shapes.shape[0]):
            shapes[i] = shapes[i] @ _kabsch(shapes[i], mean)
        new_mean = shapes.mean(axis=0)
        new_mean -= new_mean.mean(axis=0)
        if scale:
            norm = np.linalg.norm(new_mean)
            if norm > 0:
                new_mean /= norm
        change = float(np.linalg.norm(new_mean - mean))
        mean = new_mean
        if change < tol:
            break
    return shapes, mean


@dataclass
class ShapeModel:
    """A PCA shape model over ``M`` corresponded points in 3-D."""

    mean: np.ndarray  # (M, 3)
    components: np.ndarray  # (K, 3M) principal modes (unit vectors)
    eigenvalues: np.ndarray  # (K,) variance along each mode
    explained_variance_ratio: np.ndarray  # (K,)
    scores: np.ndarray  # (S, K) per-subject projections

    @property
    def n_points(self) -> int:
        return self.mean.shape[0]

    @property
    def n_modes(self) -> int:
        return self.components.shape[0]


def build_ssm(aligned: np.ndarray, n_components: int | None = None) -> ShapeModel:
    """PCA over Procrustes-aligned ``(S, M, 3)`` shapes.

    ``n_components`` defaults to (and is capped at) ``min(S - 1, 3M)`` -- the most modes the
    data can support.
    """
    shapes = np.asarray(aligned, dtype=np.float64)
    if shapes.ndim != 3 or shapes.shape[2] != 3:
        raise ValueError(f"expected (S, M, 3), got {shapes.shape}")
    n_subjects, n_points, _ = shapes.shape
    flat = shapes.reshape(n_subjects, n_points * 3)

    max_components = min(n_subjects - 1, flat.shape[1])
    k = max_components if n_components is None else min(n_components, max_components)

    pca = PCA(n_components=k)
    scores = pca.fit_transform(flat)
    return ShapeModel(
        mean=pca.mean_.reshape(n_points, 3),
        components=pca.components_,
        eigenvalues=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        scores=scores,
    )


def mode_shape(model: ShapeModel, mode: int, sd: float) -> np.ndarray:
    """The mean shape walked ``sd`` standard deviations along principal ``mode``.

    Returns an ``(M, 3)`` point set.
    """
    if not 0 <= mode < model.n_modes:
        raise IndexError(f"mode {mode} out of range [0, {model.n_modes})")
    offset = sd * np.sqrt(model.eigenvalues[mode]) * model.components[mode]
    return model.mean + offset.reshape(model.n_points, 3)
