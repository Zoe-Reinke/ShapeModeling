"""Clean segmentation masks and extract surface meshes in physical (mm) coordinates.

A raw label mask is rarely watertight: it may contain disconnected islands or interior
holes that distort a surface. :func:`clean_mask` keeps the largest connected component and
fills holes (with optional anti-alias smoothing), and :func:`extract_surface` runs marching
cubes and maps the vertices into the volume's physical frame so meshes from different scans
share a common coordinate system.

Surfaces are an inspection artefact here (the correspondence step feeds the *segmentations*
to ShapeWorks); :func:`save_ply` writes them for viewing. Uses scikit-image + scipy.ndimage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from skimage import measure

from shape_modeling.analysis.ssm import ShapeModel, mode_shape
from shape_modeling.io.reader import LoadedVolume

__all__ = [
    "CleanConfig",
    "Surface",
    "clean_mask",
    "export_mode_shapes",
    "extract_surface",
    "hull_surface",
    "save_ply",
    "warp_surface",
]


@dataclass
class CleanConfig:
    """Mask-cleaning options. Defaults suit a single solid organ."""

    keep_largest: bool = True  # drop islands, keep the largest connected component
    fill_holes: bool = True  # fill interior holes so the surface is watertight
    smooth_sigma: float = 0.0  # >0: Gaussian anti-alias then re-threshold at 0.5


@dataclass
class Surface:
    """A triangle mesh. ``verts`` are physical mm (x, y, z); ``faces`` index ``verts``."""

    verts: np.ndarray  # (N, 3) float
    faces: np.ndarray  # (M, 3) int


def clean_mask(mask: np.ndarray, config: CleanConfig | None = None) -> np.ndarray:
    """Binarise and clean a mask: largest component, hole fill, optional smoothing.

    Returns a ``uint8`` binary mask with the same shape as ``mask``.
    """
    config = config or CleanConfig()
    binary = np.asarray(mask) > 0

    if config.keep_largest and binary.any():
        labels, n = ndimage.label(binary)
        if n > 1:
            counts = np.bincount(labels.ravel())
            counts[0] = 0  # ignore background
            binary = labels == int(counts.argmax())

    if config.fill_holes:
        binary = ndimage.binary_fill_holes(binary)

    if config.smooth_sigma > 0:
        smoothed = ndimage.gaussian_filter(binary.astype(np.float32), config.smooth_sigma)
        binary = smoothed >= 0.5

    return binary.astype(np.uint8)


def _index_to_physical(verts_zyx: np.ndarray, geom: LoadedVolume) -> np.ndarray:
    """Map marching-cubes vertices (continuous z,y,x indices) to physical mm (x,y,z)."""
    if geom.sitk_image is not None:
        transform = geom.sitk_image.TransformContinuousIndexToPhysicalPoint
        # SimpleITK indexes as (x, y, z); marching cubes returns (z, y, x).
        idx_xyz = verts_zyx[:, ::-1]
        return np.array([transform(tuple(map(float, p))) for p in idx_xyz], dtype=np.float64)
    # No sitk image: fall back to spacing + origin (ignores direction cosines).
    spacing = np.asarray(geom.spacing, dtype=np.float64)  # (x, y, z)
    origin = np.asarray(geom.origin, dtype=np.float64)
    return verts_zyx[:, ::-1] * spacing + origin


def extract_surface(
    mask: np.ndarray,
    geom: LoadedVolume,
    *,
    level: float = 0.5,
) -> Surface:
    """Marching-cubes surface of a (cleaned) binary mask, in physical mm coordinates.

    ``geom`` supplies the spacing/origin/direction used to place the vertices; pass the
    :class:`LoadedVolume` the mask came from. Clean the mask first with :func:`clean_mask`.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        raise ValueError("cannot extract a surface from an empty mask")
    verts_zyx, faces, _normals, _values = measure.marching_cubes(
        binary.astype(np.float32), level=level
    )
    return Surface(verts=_index_to_physical(verts_zyx, geom), faces=faces.astype(np.int64))


def hull_surface(points: np.ndarray) -> Surface:
    """Triangulate a corresponded point set by its convex hull.

    ShapeWorks particles carry no triangle connectivity, so to *render* a particle cloud (or
    a mode-shape walked from one) we need faces. The convex hull is a reasonable closed
    triangulation for a roughly-convex organ like the prostate, and -- because particle
    ordering is shared across subjects -- the hull faces of the **mean** can be reused for
    every shape in the model (see :func:`export_mode_shapes`).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (M, 3), got {pts.shape}")
    hull = ConvexHull(pts)
    return Surface(verts=pts, faces=hull.simplices.astype(np.int64))


def _similarity_fit(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (uniform scale ``s``, rotation ``R``, translation ``t``).

    Umeyama (1991): the best ``s, R, t`` with ``dst ≈ s * (src @ R.T) + t``. Used to absorb the
    bulk frame/scale change before a thin-plate spline, so the spline only models the small
    residual deformation (a raw spline forced to do the scale change too oscillates and folds).
    """
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    corr = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # keep a proper (det=+1) rotation
        corr[-1, -1] = -1.0
    rot = u @ corr @ vt
    var_src = (src_c**2).sum() / src.shape[0]
    scale = float((d * np.diag(corr)).sum() / var_src) if var_src > 0 else 1.0
    translation = mu_dst - scale * (rot @ mu_src)
    return scale, rot, translation


def warp_surface(
    template: Surface,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.0,
) -> Surface:
    """Warp a real template mesh onto a new particle configuration (similarity + spline).

    The honest way to mesh a corresponded particle cloud: instead of inventing connectivity
    from the points (:func:`hull_surface` only spans the *convex* extreme points and bridges
    every concavity), borrow it from a **real** surface. Carry the template's particles
    (``src_points``) onto the target particles (``dst_points``) and apply the same map to every
    vertex of ``template``; faces are inherited unchanged, so concavities and full resolution
    survive, and -- because particle ordering is shared -- vertex *j* is comparable across
    whatever targets you warp to (mean, mode shapes, …).

    The map is fit in two stages: a global **similarity** (uniform scale, rotation, translation)
    that absorbs any frame/scale difference between ``src_points`` and ``dst_points``, then a
    thin-plate **spline** on the small residual. Splitting it this way is what keeps the result
    well-behaved: a single spline asked to do a large scale change as well oscillates between
    control points and folds the surface through itself. Because of the similarity stage,
    ``src_points`` and ``template`` need only share *a* frame -- it need not match ``dst_points``'.

    Typical use: ``template`` is one subject's groomed surface (marching cubes of its distance
    transform), ``src_points`` is that subject's particles in the same frame (its
    ``*_local.particles``), and ``dst_points`` is the model mean (or a walked mode shape).

    ``smoothing`` > 0 relaxes exact interpolation of the residual (useful if particles are noisy).
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be matching (M, 3); got {src.shape} and {dst.shape}")
    verts = np.asarray(template.verts, dtype=np.float64)

    # 1. global similarity src -> dst, applied to both the control points and the mesh.
    scale, rot, translation = _similarity_fit(src, dst)
    src_aligned = scale * (src @ rot.T) + translation
    verts_aligned = scale * (verts @ rot.T) + translation

    # 2. thin-plate spline on the residual only (src_aligned is already close to dst).
    warp = RBFInterpolator(src_aligned, dst, kernel=kernel, smoothing=smoothing)
    return Surface(verts=warp(verts_aligned), faces=np.asarray(template.faces))


def export_mode_shapes(
    model: ShapeModel,
    out_dir: str | Path,
    *,
    n_modes: int = 5,
    sds: tuple[float, ...] = (-3.0, -2.0, 0.0, 2.0, 3.0),
    faces: np.ndarray | None = None,
) -> list[Path]:
    """Write the mean shape walked along each mode to PLY files for 3-D inspection.

    For each of the first ``n_modes`` modes and each step in ``sds`` (standard deviations),
    writes ``<out_dir>/mode<k>_<+c>sd.ply`` -- the mean plus ``c * sqrt(eigenvalue) * mode``
    (see :func:`~shape_modeling.analysis.ssm.mode_shape`). Load the ``-3/0/+3`` files of one
    mode together in 3D Slicer or ParaView to see what variation that mode captures.

    Connectivity is the convex hull of the **mean** shape (:func:`hull_surface`), reused for
    every file so vertex ``j`` is the same point everywhere; pass ``faces`` to override.

    Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if faces is None:
        faces = hull_surface(model.mean).faces

    k = min(n_modes, model.n_modes)
    written: list[Path] = []
    for mode in range(k):
        for c in sds:
            pts = mode_shape(model, mode, c)
            path = out_dir / f"mode{mode + 1}_{c:+.0f}sd.ply"
            save_ply(Surface(verts=pts, faces=faces), path)
            written.append(path)
    return written


def save_ply(surface: Surface, path: str | Path) -> None:
    """Write a :class:`Surface` to an ASCII PLY file (for quick viewing)."""
    verts, faces = surface.verts, surface.faces
    path = Path(path)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"3 {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")
