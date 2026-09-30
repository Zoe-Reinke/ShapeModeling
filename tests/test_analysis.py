"""Tests for shape_modeling.analysis (surfaces, Procrustes, PCA, correspondence I/O).

Synthetic fixtures only, in ``(z, y, x)`` order like ``test_qc.py``. The ShapeWorks
optimizer itself is exercised only when the package is importable (opt-in, like the
``SHAPEMODELING_DATA_DIR`` real-data test in ``test_reader.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.spatial import ConvexHull

from shape_modeling.analysis import (
    CleanConfig,
    ExportedModel,
    ShapeModel,
    build_ssm,
    centroid_size,
    clean_mask,
    combat_harmonize,
    compactness,
    evaluate_ssm,
    export_mode_shapes,
    export_model,
    extract_surface,
    generalization,
    gpa,
    hull_surface,
    load_model,
    load_shapeworks_evaluation,
    mode_shape,
    qc_mesh,
    reconstruct_surface,
    save_ply,
    smooth_surface,
    specificity,
    surface_distance,
    warp_surface,
)
from shape_modeling.analysis.cohort import CohortEntry
from shape_modeling.analysis.correspondence import (
    ParticleSystem,
    ShapeWorksConfig,
    _check_particles,
    load_particles,
    particle_spread_ratio,
)
from shape_modeling.analysis.evaluation import SSMEvaluation, default_mode_grid
from shape_modeling.analysis.surface import Surface, _similarity_fit
from shape_modeling.io.reader import LoadedVolume, VolumeType

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ellipsoid(shape: tuple[int, int, int], radii: tuple[float, float, float]) -> np.ndarray:
    """A solid centred ellipsoid in a (z, y, x) array."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = (s / 2 for s in shape)
    rz, ry, rx = radii
    norm = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return (norm <= 1.0).astype(np.uint8)


def _loaded_mask(array: np.ndarray, spacing: tuple[float, float, float]) -> LoadedVolume:
    """Wrap a (z, y, x) array as a LoadedVolume with a matching SimpleITK image."""
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(spacing)  # SimpleITK spacing is (x, y, z)
    return LoadedVolume(
        path=Path("synthetic.nrrd"),
        volume_type=VolumeType.ORGAN_SEG,
        size=tuple(img.GetSize()),
        spacing=spacing,
        origin=img.GetOrigin(),
        direction=img.GetDirection(),
        dtype="uint8",
        sitk_image=img,
        array=array,
    )


# --------------------------------------------------------------------------- #
# clean_mask
# --------------------------------------------------------------------------- #


def test_clean_mask_keeps_largest_fills_holes_drops_island() -> None:
    mask = _ellipsoid((24, 24, 24), (8, 8, 8))
    mask[12, 12, 12] = 0  # punch an interior hole
    mask[0:2, 0:2, 0:2] = 1  # a small disconnected island in the corner

    cleaned = clean_mask(mask, CleanConfig())

    assert cleaned[0, 0, 0] == 0  # island removed
    assert cleaned[12, 12, 12] == 1  # hole filled
    # Exactly one connected component remains.
    from scipy import ndimage

    _, n = ndimage.label(cleaned)
    assert n == 1


def test_clean_mask_no_fill_keeps_hole() -> None:
    mask = _ellipsoid((24, 24, 24), (8, 8, 8))
    mask[12, 12, 12] = 0
    cleaned = clean_mask(mask, CleanConfig(fill_holes=False))
    assert cleaned[12, 12, 12] == 0


# --------------------------------------------------------------------------- #
# extract_surface
# --------------------------------------------------------------------------- #


def test_extract_surface_is_closed_and_in_mm() -> None:
    spacing = (2.0, 2.0, 2.0)  # anisotropic-friendly: easy to check scaling
    array = _ellipsoid((30, 30, 30), (10, 10, 10))
    vol = _loaded_mask(array, spacing)

    surf = extract_surface(array, vol)

    assert surf.verts.shape[1] == 3
    assert surf.faces.shape[1] == 3
    # A closed triangulated surface: every edge is shared by two triangles, so 2E = 3F.
    from collections import Counter

    edges: Counter[tuple[int, int]] = Counter()
    for a, b, c in surf.faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges[(min(u, v), max(u, v))] += 1
    assert all(count == 2 for count in edges.values())
    # Physical: a radius-10-voxel ellipsoid at 2 mm/voxel spans ~40 mm, not ~20.
    extent = surf.verts.max(axis=0) - surf.verts.min(axis=0)
    assert np.all(extent > 30.0)


def test_extract_surface_fallback_without_sitk_image() -> None:
    spacing = (1.5, 1.5, 1.5)
    array = _ellipsoid((20, 20, 20), (6, 6, 6))
    vol = _loaded_mask(array, spacing)
    vol.sitk_image = None  # force the spacing-only fallback path
    surf = extract_surface(array, vol)
    extent = surf.verts.max(axis=0) - surf.verts.min(axis=0)
    assert np.all(extent > 12.0)


def test_extract_surface_empty_raises() -> None:
    vol = _loaded_mask(np.zeros((10, 10, 10), np.uint8), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="empty mask"):
        extract_surface(vol.array, vol)


def test_save_ply_roundtrip_counts(tmp_path: Path) -> None:
    array = _ellipsoid((20, 20, 20), (6, 6, 6))
    vol = _loaded_mask(array, (1.0, 1.0, 1.0))
    surf = extract_surface(array, vol)
    out = tmp_path / "mesh.ply"
    save_ply(surf, out)
    text = out.read_text()
    assert f"element vertex {len(surf.verts)}" in text
    assert f"element face {len(surf.faces)}" in text


# --------------------------------------------------------------------------- #
# Generalized Procrustes alignment
# --------------------------------------------------------------------------- #


def _rotation(angles: tuple[float, float, float]) -> np.ndarray:
    ax, ay, az = angles
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_gpa_removes_pose_and_scale() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(12, 3))  # an asymmetric point cloud
    transforms = [
        (1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        (2.5, (0.3, -0.4, 0.7), (5.0, -3.0, 2.0)),
        (0.4, (-0.6, 0.2, -0.1), (-1.0, 4.0, -2.0)),
    ]
    shapes = np.stack(
        [scale * (base @ _rotation(ang).T) + np.array(trans) for scale, ang, trans in transforms]
    )

    aligned, mean = gpa(shapes, scale=True)

    # All shapes are the same up to similarity -> aligned copies coincide.
    for i in range(aligned.shape[0]):
        assert np.allclose(aligned[i], aligned[0], atol=1e-6)
    assert np.allclose(mean.mean(axis=0), 0.0, atol=1e-6)  # mean is centred


def test_gpa_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(S, M, 3\)"):
        gpa(np.zeros((3, 5, 2)))


# --------------------------------------------------------------------------- #
# PCA shape model
# --------------------------------------------------------------------------- #


def test_build_ssm_recovers_single_mode() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(15, 3))
    delta = rng.normal(size=(15, 3))  # one fixed direction of variation
    alphas = np.linspace(-2.0, 2.0, 7)
    shapes = np.stack([base + a * delta for a in alphas])

    model = build_ssm(shapes)

    assert isinstance(model, ShapeModel)
    assert model.explained_variance_ratio[0] > 0.99  # one dominant mode
    # Eigenvalues are sorted descending.
    assert np.all(np.diff(model.eigenvalues) <= 1e-9)
    # Mode-1 score tracks the generating parameter (monotonic, |corr| ~ 1).
    corr = np.corrcoef(model.scores[:, 0], alphas)[0, 1]
    assert abs(corr) > 0.999


def test_build_ssm_caps_components_at_n_minus_one() -> None:
    rng = np.random.default_rng(2)
    shapes = rng.normal(size=(5, 10, 3))
    model = build_ssm(shapes, n_components=10)
    assert model.n_modes == 4  # min(S-1, 3M) = min(4, 30)


def test_mode_shape_walks_the_mean() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=(10, 3))
    delta = rng.normal(size=(10, 3))
    shapes = np.stack([base + a * delta for a in np.linspace(-1, 1, 6)])
    model = build_ssm(shapes)

    at_zero = mode_shape(model, 0, 0.0)
    assert np.allclose(at_zero, model.mean)
    plus = mode_shape(model, 0, 2.0)
    minus = mode_shape(model, 0, -2.0)
    # +2sd and -2sd are symmetric about the mean.
    assert np.allclose(plus + minus, 2 * model.mean, atol=1e-9)
    with pytest.raises(IndexError):
        mode_shape(model, model.n_modes, 1.0)


def test_centroid_size_scalar_and_stack() -> None:
    # A unit cube centred at the origin: 8 corners, each at distance sqrt(3)/2 from centroid.
    cube = np.array(list(np.ndindex(2, 2, 2)), dtype=float) - 0.5
    expected = np.sqrt(8 * (3 * 0.25))  # sum of squared distances = 8 * (3 * 0.5^2)
    assert np.isclose(centroid_size(cube), expected)

    # A stack returns one size per shape, invariant to translation, scaling with shape size.
    stack = np.stack([cube, cube + 100.0, cube * 2.0])
    sizes = centroid_size(stack)
    assert sizes.shape == (3,)
    assert np.isclose(sizes[0], sizes[1])  # translation-invariant
    assert np.isclose(sizes[2], 2 * sizes[0])  # scales linearly


def test_centroid_size_constant_after_gpa_scaling() -> None:
    rng = np.random.default_rng(5)
    shapes = rng.normal(size=(6, 12, 3)) * rng.uniform(1, 5, size=(6, 1, 1))  # varied sizes
    aligned, _ = gpa(shapes, scale=True)
    sizes = centroid_size(aligned)
    # scale=True normalises every shape, so aligned centroid sizes collapse to ~constant.
    assert np.allclose(sizes, sizes[0])


def test_hull_surface_closed_triangulation() -> None:
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(50, 3))
    surf = hull_surface(pts)
    assert surf.verts.shape == (50, 3)
    assert surf.faces.shape[1] == 3
    # A closed triangulated surface satisfies Euler's formula F = 2V - 4 (genus 0, all tris).
    n_hull_verts = len(np.unique(surf.faces))
    assert surf.faces.shape[0] == 2 * n_hull_verts - 4


def _union_of_spheres(n: int = 500, offset: float = 0.9) -> np.ndarray:
    """Surface samples of two overlapping unit spheres -- a waist a convex hull bridges over."""
    rng = np.random.default_rng(3)
    centres = np.array([[-offset, 0.0, 0.0], [offset, 0.0, 0.0]])
    halves = []
    for own, other in (centres, centres[::-1]):
        dirs = rng.normal(size=(4 * n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        on_sphere = dirs + own
        halves.append(on_sphere[np.linalg.norm(on_sphere - other, axis=1) > 1.0][: n // 2])
    return np.concatenate(halves)


def _enclosed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Divergence-theorem volume; assumes a closed, consistently wound mesh."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def test_reconstruct_surface_uses_every_point_and_is_watertight() -> None:
    pts = _union_of_spheres()
    surf = reconstruct_surface(pts)

    assert np.array_equal(surf.verts, pts)  # vertices are the input points, in order
    assert surf.faces.max() < len(pts)
    assert len(np.unique(surf.faces)) == len(pts)  # every correspondence point is on the mesh
    assert len(surf.faces) == 2 * len(pts) - 4  # closed genus-0 triangulation (Euler)

    report = qc_mesh(surf)
    assert report.ok
    assert report.is_watertight and report.is_edge_manifold and report.is_vertex_manifold
    assert report.is_consistently_oriented and not report.is_self_intersecting
    assert report.n_unreferenced_vertices == 0


def test_reconstruct_surface_follows_a_concavity_the_hull_bridges() -> None:
    pts = _union_of_spheres()
    rec = reconstruct_surface(pts)

    # Two unit spheres at +-0.9 enclose 8.33 units^3 (spheres minus their lens); a mesh through
    # surface samples is inscribed, so it lands just under. The convex hull fills the waist in.
    rec_volume = _enclosed_volume(rec.verts, rec.faces)
    assert 0.93 * 8.33 < rec_volume < 8.33
    assert ConvexHull(pts).volume > 1.13 * rec_volume

    # The hull also throws points away -- the defect that made the exported mesh unusable.
    hull_report = qc_mesh(hull_surface(pts))
    assert hull_report.n_unreferenced_vertices > 0
    assert not hull_report.ok


def test_reconstruct_surface_topology_transfers_to_another_shape() -> None:
    """The point of a fixed face list: it stays valid when the SSM moves the vertices."""
    pts = _union_of_spheres()
    faces = reconstruct_surface(pts).faces
    stretched = pts * np.array([1.4, 0.9, 1.0])  # a plausible mode-like deformation

    report = qc_mesh(Surface(verts=stretched, faces=faces))
    assert report.ok
    assert np.array_equal(faces, reconstruct_surface(pts).faces)  # deterministic


def test_qc_mesh_detects_a_fold_and_a_flipped_face() -> None:
    pts = _union_of_spheres(n=250)
    surf = reconstruct_surface(pts)

    folded = surf.verts.copy()
    folded[0] = -1.5 * folded[0]  # drag one vertex through the opposite wall
    assert qc_mesh(Surface(verts=folded, faces=surf.faces)).is_self_intersecting

    flipped = surf.faces.copy()
    flipped[0] = flipped[0][::-1]
    report = qc_mesh(Surface(verts=surf.verts, faces=flipped))
    assert not report.is_consistently_oriented  # winding no longer agrees across the edge
    assert report.is_orientable  # ...but it can be rewound: the surface itself is fine


def test_reconstruct_surface_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="expected"):
        reconstruct_surface(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="at least 4 points"):
        reconstruct_surface(np.zeros((3, 3)))
    # A flat (or collapsed) cloud has no interior to carve -- refused with a readable error
    # rather than a raw QhullError, and never silently meshed the way a hull would be.
    flat = np.random.default_rng(0).normal(size=(40, 3)) * np.array([1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="degenerate"):
        reconstruct_surface(flat)

def test_warp_surface_keeps_faces_and_lands_on_target() -> None:
    rng = np.random.default_rng(11)
    # A template mesh (cube corners + a face triangulation) and its "particles".
    verts = rng.normal(size=(30, 3))
    faces = hull_surface(verts).faces
    src = rng.normal(size=(12, 3))
    # A known affine target so we can check the warp reproduces it exactly at the control points.
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    dst = src @ rot * 2.0 + np.array([5.0, -1.0, 3.0])

    warped = warp_surface(Surface(verts=verts, faces=faces), src, dst)
    # connectivity is inherited unchanged...
    assert np.array_equal(warped.faces, faces)
    assert warped.verts.shape == verts.shape
    # ...and the warp interpolates: evaluating it at the source particles returns the targets.
    control = warp_surface(Surface(verts=src, faces=np.empty((0, 3), int)), src, dst)
    np.testing.assert_allclose(control.verts, dst, atol=1e-8)


def _mean_edge_roughness(verts: np.ndarray, faces: np.ndarray) -> float:
    """Mean distance of each vertex from the average of its edge-neighbours (bump size)."""
    n = len(verts)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    nbr_sum = np.zeros((n, 3))
    np.add.at(nbr_sum, i, verts[j])
    np.add.at(nbr_sum, j, verts[i])
    deg = np.bincount(np.concatenate([i, j]), minlength=n).clip(min=1)
    return float(np.linalg.norm(verts - nbr_sum / deg[:, None], axis=1).mean())


def test_smooth_surface_reduces_roughness_keeps_faces_and_shape() -> None:
    # A smooth sphere (convex hull of points on a sphere), then roughened with radial noise.
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(400, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    faces = hull_surface(dirs).faces
    noisy = dirs * (1.0 + rng.normal(scale=0.05, size=(len(dirs), 1)))  # radial bumps

    smoothed = smooth_surface(Surface(verts=noisy, faces=faces))

    # faces are untouched, vertex count preserved
    assert np.array_equal(smoothed.faces, faces)
    assert smoothed.verts.shape == noisy.shape
    # high-frequency roughness drops sharply...
    before = _mean_edge_roughness(noisy, faces)
    after = _mean_edge_roughness(smoothed.verts, faces)
    assert after < before / 2
    # ...while the gross shape (overall radius) is preserved (Taubin does not shrink like Laplacian)
    r_before = np.linalg.norm(noisy - noisy.mean(0), axis=1).mean()
    r_after = np.linalg.norm(smoothed.verts - smoothed.verts.mean(0), axis=1).mean()
    assert abs(r_after - r_before) / r_before < 0.1


def test_warp_surface_rejects_mismatched_point_sets() -> None:
    mesh = Surface(verts=np.zeros((4, 3)), faces=np.empty((0, 3), int))
    with pytest.raises(ValueError, match="matching"):
        warp_surface(mesh, np.zeros((5, 3)), np.zeros((6, 3)))


def test_export_mode_shapes_writes_ply_per_step(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    base = rng.normal(size=(20, 3))
    delta = rng.normal(size=(20, 3))
    shapes = np.stack([base + a * delta for a in np.linspace(-1, 1, 8)])
    model = build_ssm(shapes)

    sds = (-3.0, 0.0, 3.0)
    paths = export_mode_shapes(model, tmp_path, n_modes=2, sds=sds)
    assert len(paths) == 2 * len(sds)
    assert (tmp_path / "mode1_+0sd.ply").exists()
    assert (tmp_path / "mode2_+3sd.ply").exists()
    # every file shares the mean's hull connectivity (same face count).
    n_faces = len(hull_surface(model.mean).faces)
    for p in paths:
        assert f"element face {n_faces}" in p.read_text()


# --------------------------------------------------------------------------- #
# ComBat harmonization
# --------------------------------------------------------------------------- #


def test_combat_harmonize_validates_shapes() -> None:
    # These guards run before the deferred neuroCombat import, so no install needed.
    with pytest.raises(ValueError, match=r"\(S, M, 3\)"):
        combat_harmonize(np.zeros((4, 5, 2)), np.array(["a", "b", "a", "b"]))
    with pytest.raises(ValueError, match="batch must be"):
        combat_harmonize(np.zeros((4, 5, 3)), np.array(["a", "b", "a"]))
    with pytest.raises(ValueError, match="2 batch levels"):
        combat_harmonize(np.zeros((4, 5, 3)), np.array(["a", "a", "a", "a"]))


def test_combat_harmonize_removes_site_keeps_size() -> None:
    neurocombat = pytest.importorskip("neuroCombat", reason="neuroCombat not installed")
    assert neurocombat  # silence unused-import linters; the import itself is the gate
    rng = np.random.default_rng(17)
    n_points = 40
    base = rng.normal(size=(n_points, 3))
    direction = rng.normal(size=(n_points, 3))  # a genuine "size"/biology axis

    coords, batch, sizes = [], [], []
    # Two sites, each with an additive coordinate shift that ComBat should remove. Real size
    # variation (along `direction`) is shared across sites and must survive.
    for site, shift in (("siteA", -1.5), ("siteB", +1.5)):
        for _ in range(20):
            size = rng.normal()
            shape = base + size * direction + shift + rng.normal(scale=0.05, size=(n_points, 3))
            coords.append(shape)
            batch.append(site)
            sizes.append(size)
    coords = np.stack(coords)
    batch = np.array(batch)
    sizes = np.array(sizes)

    harmonized = combat_harmonize(coords, batch, sizes, covariate_names=["size"])
    assert harmonized.shape == coords.shape

    # Per-site mean separation (the batch effect) shrinks dramatically.
    def site_gap(arr: np.ndarray) -> float:
        a = arr[batch == "siteA"].mean(axis=0)
        b = arr[batch == "siteB"].mean(axis=0)
        return float(np.linalg.norm(a - b))

    assert site_gap(harmonized) < 0.25 * site_gap(coords)

    # The biological size axis is preserved: projection onto `direction` still tracks the
    # generating size parameter about as well as before harmonization.
    unit = (direction / np.linalg.norm(direction)).ravel()
    proj_before = coords.reshape(len(coords), -1) @ unit
    proj_after = harmonized.reshape(len(harmonized), -1) @ unit
    r_before = abs(np.corrcoef(proj_before, sizes)[0, 1])
    r_after = abs(np.corrcoef(proj_after, sizes)[0, 1])
    assert r_after > 0.9 * r_before


# --------------------------------------------------------------------------- #
# Correspondence I/O (no ShapeWorks needed)
# --------------------------------------------------------------------------- #


def test_load_particles_reads_files(tmp_path: Path) -> None:
    case_ids = ["TestData/Case00", "Prostate158/020"]  # namespaced ids contain "/"
    pts = {cid: np.arange(15, dtype=float).reshape(5, 3) + i for i, cid in enumerate(case_ids)}
    for cid, arr in pts.items():
        # files are written with sanitized stems (no slashes)
        np.savetxt(tmp_path / f"{cid.replace('/', '__')}_world.particles", arr)

    system = load_particles(tmp_path, case_ids)
    assert system.case_ids == case_ids
    assert system.particles.shape == (2, 5, 3)
    assert np.allclose(system.particles[1], pts[case_ids[1]])


def test_load_particles_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_particles(tmp_path, ["nope"])


def test_particle_system_save_load(tmp_path: Path) -> None:
    particles = np.random.default_rng(4).normal(size=(2, 8, 3))
    system = ParticleSystem(case_ids=["a", "b"], particles=particles)
    out = tmp_path / "ps.npz"
    system.save(out)
    loaded = ParticleSystem.load(out)
    assert loaded.case_ids == system.case_ids
    assert np.allclose(loaded.particles, system.particles)


def test_shapeworks_config_defaults() -> None:
    cfg = ShapeWorksConfig()
    assert cfg.num_particles == 512
    assert cfg.use_cache is True


def _entry_with_mask(array: np.ndarray) -> CohortEntry:
    vol = _loaded_mask(array, (1.0, 1.0, 1.0))
    return CohortEntry(case_id="c", source="s", volume_cm3=1.0, organ=vol)


def test_particle_spread_ratio_compact_vs_scattered() -> None:
    # A ~20 mm organ (ellipsoid radii 10 voxels at 1 mm => ~20 mm across).
    entry = _entry_with_mask(_ellipsoid((30, 30, 30), (10, 10, 10)))
    rng = np.random.default_rng(3)
    # Particles hugging the organ: spread ~ organ extent -> ratio ~1.
    compact = ParticleSystem(case_ids=["c"], particles=rng.uniform(-10, 10, size=(1, 200, 3)))
    # Particles scattered across a 5x-larger box -> ratio ~5.
    scattered = ParticleSystem(case_ids=["c"], particles=rng.uniform(-50, 50, size=(1, 200, 3)))
    assert particle_spread_ratio(compact, [entry]) < 1.75
    assert particle_spread_ratio(scattered, [entry]) > 3.0


def test_check_particles_rejects_scattered_and_respects_disable() -> None:
    entry = _entry_with_mask(_ellipsoid((30, 30, 30), (10, 10, 10)))
    scattered = ParticleSystem(
        case_ids=["c"], particles=np.random.default_rng(5).uniform(-50, 50, size=(1, 200, 3))
    )
    with pytest.raises(ValueError, match="degenerate"):
        _check_particles(scattered, [entry], ShapeWorksConfig())
    # None disables the gate.
    _check_particles(scattered, [entry], ShapeWorksConfig(max_spread_ratio=None))


# --------------------------------------------------------------------------- #
# Opt-in: real ShapeWorks optimizer (skips unless installed)
# --------------------------------------------------------------------------- #


def test_run_shapeworks_smoke(tmp_path: Path) -> None:
    pytest.importorskip("shapeworks", reason="ShapeWorks not installed")
    from shape_modeling.analysis.cohort import CohortEntry
    from shape_modeling.analysis.correspondence import run_shapeworks

    cohort = []
    for i, cid in enumerate(["s0", "s1", "s2"]):
        array = _ellipsoid((24, 24, 24), (8 + i, 8, 8))  # slightly varying glands
        vol = _loaded_mask(array, (1.0, 1.0, 1.0))
        cohort.append(CohortEntry(case_id=cid, source="synthetic", volume_cm3=1.0, organ=vol))

    config = ShapeWorksConfig(num_particles=32, iterations=50, work_dir=tmp_path / "sw")
    system = run_shapeworks(cohort, config)
    assert system.case_ids == ["s0", "s1", "s2"]
    assert system.particles.shape[0] == 3
    assert system.particles.shape[2] == 3


# --------------------------------------------------------------------------- #
# Model export (stage 7)
# --------------------------------------------------------------------------- #


def _single_mode_model() -> ShapeModel:
    """A small SSM dominated by one mode (reuses the single-mode fixture pattern)."""
    rng = np.random.default_rng(7)
    base = rng.normal(size=(12, 3))
    delta = rng.normal(size=(12, 3))
    shapes = np.stack([base + a * delta for a in np.linspace(-2.0, 2.0, 9)])
    return build_ssm(shapes)


def _two_mode_model() -> ShapeModel:
    """An SSM with two well-separated, non-degenerate modes (for pose + multi-mode fits)."""
    rng = np.random.default_rng(8)
    base = rng.normal(size=(12, 3))
    d0 = rng.normal(size=(12, 3))
    d1 = rng.normal(size=(12, 3))
    a0 = rng.normal(size=20)
    a1 = 0.5 * rng.normal(size=20)  # second mode genuinely present, smaller variance
    shapes = np.stack([base + a0[i] * d0 + a1[i] * d1 for i in range(20)])
    return build_ssm(shapes)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` roughly-even points on the unit sphere (a dense, surface-like point set)."""
    i = np.arange(n)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def _ellipsoid_model() -> ShapeModel:
    """A surface SSM: a unit sphere whose two leading modes stretch the x and y axes.

    Unlike the abstract point-cloud fixtures, the points lie on a real surface, so nearest-point
    matching (ICP, used by ``fit_surface``) is well-posed -- this mirrors a real organ particle set.
    """
    sphere = _fibonacci_sphere(200)
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=30), rng.normal(size=30)
    shapes = np.stack(
        [sphere * np.array([1.0 + 0.3 * a[i], 1.0 + 0.2 * b[i], 1.0]) for i in range(30)]
    )
    return build_ssm(shapes)


def load_model_from(model: ShapeModel) -> ExportedModel:
    """Wrap a ShapeModel as an ExportedModel in-memory (no disk round-trip)."""
    return ExportedModel(
        mean=model.mean,
        components=model.components,
        eigenvalues=model.eigenvalues,
        explained_variance_ratio=model.explained_variance_ratio,
    )


def test_export_model_roundtrip_and_metadata(tmp_path: Path) -> None:
    model = _single_mode_model()
    faces = hull_surface(model.mean).faces
    path = export_model(
        model, tmp_path / "ssm", faces=faces, metadata={"cohort": "synthetic", "scale": True}
    )
    assert path.suffix == ".npz"

    loaded = load_model(path)
    assert loaded.n_points == model.n_points
    assert loaded.n_modes == model.n_modes
    np.testing.assert_allclose(loaded.mean, model.mean)
    np.testing.assert_allclose(loaded.components, model.components)
    np.testing.assert_allclose(loaded.eigenvalues, model.eigenvalues)
    np.testing.assert_array_equal(loaded.faces, faces)
    assert loaded.metadata["cohort"] == "synthetic"
    assert loaded.metadata["units"] == "normalized"


def test_export_model_without_faces(tmp_path: Path) -> None:
    model = _single_mode_model()
    loaded = load_model(export_model(model, tmp_path / "ssm"))
    assert loaded.faces is None
    with pytest.raises(ValueError, match="no faces"):
        loaded.surface(np.zeros(loaded.n_modes))


def test_instance_mean_and_mode_walk() -> None:
    model = _single_mode_model()
    loaded = load_model_from(model)

    # weights=None and all-zero weights both return the mean.
    np.testing.assert_allclose(loaded.instance(), model.mean)
    np.testing.assert_allclose(loaded.instance(np.zeros(loaded.n_modes)), model.mean)

    # +/- along mode 0 are symmetric about the mean; matches mode_shape().
    plus = loaded.instance([2.0])
    np.testing.assert_allclose(plus, mode_shape(model, 0, 2.0))
    minus = loaded.instance([-2.0])
    np.testing.assert_allclose(plus + minus, 2 * model.mean, atol=1e-9)

    with pytest.raises(ValueError, match="only"):
        loaded.instance(np.zeros(loaded.n_modes + 1))


def test_instance_applies_pose() -> None:
    model = _single_mode_model()
    loaded = load_model_from(model)
    rot = _rotation((0.3, -0.2, 0.5))
    posed = loaded.instance([1.0], pose=(2.0, rot, np.array([10.0, -5.0, 3.0])))
    expected = 2.0 * (loaded.instance([1.0]) @ rot.T) + np.array([10.0, -5.0, 3.0])
    np.testing.assert_allclose(posed, expected)


def test_fit_recovers_weights_in_model_frame() -> None:
    model = _single_mode_model()
    loaded = load_model_from(model)
    target = loaded.instance([1.5])  # a known shape, already in the model frame

    result = loaded.fit(target, pose=False)

    assert result.weights[0] == pytest.approx(1.5, abs=1e-6)
    assert result.rmse < 1e-9
    np.testing.assert_allclose(result.points, target, atol=1e-9)


def test_fit_recovers_pose_and_shape() -> None:
    model = _two_mode_model()
    loaded = load_model_from(model)
    rot = _rotation((0.2, 0.4, -0.3))
    pose = (1.7, rot, np.array([4.0, 8.0, -2.0]))
    target = loaded.instance([1.0, -0.5], pose=pose)

    result = loaded.fit(target, n_modes=2, n_iter=30)

    assert result.scale == pytest.approx(1.7, rel=1e-2)
    assert result.weights[0] == pytest.approx(1.0, abs=1e-2)
    assert result.weights[1] == pytest.approx(-0.5, abs=1e-2)
    assert result.rmse < 1e-4
    np.testing.assert_allclose(result.points, target, atol=1e-3)


def test_fit_clamps_to_plausible_box() -> None:
    model = _single_mode_model()
    loaded = load_model_from(model)
    target = loaded.instance([5.0])  # implausibly far along mode 0

    result = loaded.fit(target, pose=False, max_sd=3.0)
    assert result.weights[0] == pytest.approx(3.0, abs=1e-6)


def test_fit_rejects_uncorresponded_target() -> None:
    model = _single_mode_model()
    loaded = load_model_from(model)
    with pytest.raises(ValueError, match="correspondence"):
        loaded.fit(np.zeros((loaded.n_points + 1, 3)))


def test_surface_distance_zero_for_identical_and_positive_otherwise() -> None:
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(50, 3))
    mean, rms, hd95 = surface_distance(cloud, cloud)
    assert mean == pytest.approx(0.0)
    assert rms == pytest.approx(0.0)
    assert hd95 == pytest.approx(0.0)

    # A pure 1 mm shift puts every nearest-neighbour distance at ~1 (clouds are dense enough).
    shifted_mean, shifted_rms, _ = surface_distance(cloud, cloud + np.array([1.0, 0.0, 0.0]))
    assert shifted_mean > 0
    assert shifted_rms >= shifted_mean  # RMS >= mean always


def test_fit_surface_recovers_shape_from_uncorresponded_cloud() -> None:
    loaded = load_model_from(_ellipsoid_model())
    rot = _rotation((0.2, -0.3, 0.4))
    pose = (1.5, rot, np.array([5.0, -2.0, 7.0]))
    truth = loaded.instance([1.5, -1.0], pose=pose)  # an anisotropic, posed held-out ellipsoid

    # Break correspondence: shuffle the rows and append duplicates so N != M and ordering is gone.
    rng = np.random.default_rng(3)
    shuffled = truth[rng.permutation(len(truth))]
    target = np.vstack([shuffled, shuffled[:20]])  # every point still lies on the true surface

    result = loaded.fit_surface(target, n_modes=2, n_iter=100)

    assert result.mean_distance < 0.05  # near-perfect reconstruction (sphere radius ~1)
    assert result.scale == pytest.approx(1.5, rel=0.05)
    assert result.weights[0] == pytest.approx(1.5, abs=0.1)
    assert result.weights[1] == pytest.approx(-1.0, abs=0.1)
    assert result.points.shape == loaded.mean.shape


def test_fit_surface_accepts_surface_and_honours_init_pose() -> None:
    loaded = load_model_from(_ellipsoid_model())
    truth = loaded.instance([1.5, -1.0])  # anisotropic, already in the model frame
    surf = hull_surface(truth)  # a Surface input (verts on the true shape)

    # An identity init_pose skips the multi-restart search; the model frame already matches.
    result = loaded.fit_surface(surf, n_modes=2, init_pose=(1.0, np.eye(3), np.zeros(3)))
    assert result.mean_distance < 0.05
    assert result.weights[0] == pytest.approx(1.5, abs=0.1)
    assert result.weights[1] == pytest.approx(-1.0, abs=0.1)


def _mm_model(model: ShapeModel) -> ExportedModel:
    """Wrap a ShapeModel as an ExportedModel that declares millimetre units."""
    return ExportedModel(
        mean=model.mean,
        components=model.components,
        eigenvalues=model.eigenvalues,
        explained_variance_ratio=model.explained_variance_ratio,
        metadata={"units": "mm"},
    )


def test_units_defaults_to_normalized_and_selects_the_pose_family() -> None:
    model = _ellipsoid_model()
    assert load_model_from(model).units == "normalized"  # no metadata -> legacy default
    assert _mm_model(model).units == "mm"


def test_export_model_records_mm_units(tmp_path: Path) -> None:
    loaded = load_model(export_model(_single_mode_model(), tmp_path / "ssm", units="mm"))
    assert loaded.units == "mm"
    assert loaded.metadata["units"] == "mm"


def test_mm_model_fits_rigidly_and_keeps_its_own_size() -> None:
    """A "mm" model solves 6-DOF: it recovers a rigidly posed target without rescaling."""
    loaded = _mm_model(_ellipsoid_model())
    rot = _rotation((0.2, -0.3, 0.4))
    truth = loaded.instance([1.5, -1.0], pose=(1.0, rot, np.array([5.0, -2.0, 7.0])))

    result = loaded.fit_surface(truth, n_modes=2, n_iter=100)

    assert result.scale == 1.0  # pinned exactly, not merely close
    assert result.mean_distance < 0.05
    assert result.weights[0] == pytest.approx(1.5, abs=0.1)
    assert result.weights[1] == pytest.approx(-1.0, abs=0.1)


def test_mm_model_refuses_to_rescale_onto_a_resized_target() -> None:
    """The size prior bites: a 1.5x target is not absorbed by the pose, unlike a unitless model."""
    model = _ellipsoid_model()
    truth = load_model_from(model).instance([1.0, -0.5], pose=(1.5, np.eye(3), np.zeros(3)))

    rigid = _mm_model(model).fit_surface(truth, n_modes=2, n_iter=100)
    similarity = load_model_from(model).fit_surface(truth, n_modes=2, n_iter=100)

    assert rigid.scale == 1.0
    assert similarity.scale == pytest.approx(1.5, rel=0.05)  # unitless model swallows the size
    assert rigid.mean_distance > similarity.mean_distance  # the mm model pays for the size gap


def test_fit_scale_overrides_the_units_default() -> None:
    model = _ellipsoid_model()
    truth = load_model_from(model).instance([1.0, -0.5], pose=(1.5, np.eye(3), np.zeros(3)))

    forced = _mm_model(model).fit_surface(truth, n_modes=2, n_iter=100, fit_scale=True)
    pinned = load_model_from(model).fit_surface(truth, n_modes=2, n_iter=100, fit_scale=False)

    assert forced.scale == pytest.approx(1.5, rel=0.05)  # mm model told to solve scale anyway
    assert pinned.scale == 1.0  # unitless model told not to


def test_corresponded_fit_honours_rigid_units() -> None:
    loaded = _mm_model(_two_mode_model())
    rot = _rotation((0.2, 0.4, -0.3))
    target = loaded.instance([1.0, -0.5], pose=(1.0, rot, np.array([4.0, 8.0, -2.0])))

    result = loaded.fit(target, n_modes=2, n_iter=30)

    assert result.scale == 1.0
    assert result.weights[0] == pytest.approx(1.0, abs=1e-2)
    assert result.rmse < 1e-4


def test_rigid_similarity_fit_is_the_umeyama_rotation_without_scale() -> None:
    rng = np.random.default_rng(5)
    src = rng.normal(size=(40, 3))
    rot = _rotation((0.3, -0.1, 0.2))
    dst = 2.0 * (src @ rot.T) + np.array([1.0, 2.0, 3.0])  # a deliberately rescaled target

    scale, r, t = _similarity_fit(src, dst, allow_scale=False)

    assert scale == 1.0
    np.testing.assert_allclose(r, rot, atol=1e-9)  # rotation is unaffected by the pinned scale
    np.testing.assert_allclose(t, dst.mean(0) - (r @ src.mean(0)), atol=1e-9)


# --------------------------------------------------------------------------- #
# SSM quality metrics (compactness / generalization / specificity)
# --------------------------------------------------------------------------- #


def _two_mode_population(rng: np.random.Generator, n: int = 30, m: int = 20) -> np.ndarray:
    """A population with exactly two dominant, ordered modes plus tiny noise (in mm)."""
    base = rng.normal(size=(m, 3)) * 10.0
    mode_a = rng.normal(size=(m, 3))
    mode_b = rng.normal(size=(m, 3))
    a = rng.normal(scale=4.0, size=n)  # dominant
    b = rng.normal(scale=1.0, size=n)  # secondary
    shapes = base[None] + a[:, None, None] * mode_a[None] + b[:, None, None] * mode_b[None]
    return shapes + rng.normal(scale=0.01, size=shapes.shape)


def test_compactness_matches_pca_variance_ratio_and_is_monotone() -> None:
    rng = np.random.default_rng(0)
    shapes = _two_mode_population(rng)
    cum = compactness(shapes)
    # equals the cumulative explained-variance ratio of a PCA on the same shapes
    flat = shapes.reshape(shapes.shape[0], -1)
    pca_cum = np.cumsum(
        np.linalg.svd(flat - flat.mean(0), compute_uv=False) ** 2
    )
    pca_cum = pca_cum / pca_cum[-1]
    np.testing.assert_allclose(cum, pca_cum[: len(cum)], atol=1e-9)
    assert np.all(np.diff(cum) >= -1e-12)  # non-decreasing
    assert cum[-1] == pytest.approx(1.0, abs=1e-9)
    # two dominant modes -> first two capture the bulk of the variance
    assert cum[1] > 0.95


def test_compactness_is_scale_invariant() -> None:
    rng = np.random.default_rng(1)
    shapes = _two_mode_population(rng)
    np.testing.assert_allclose(compactness(shapes), compactness(shapes * 7.3), atol=1e-9)


def test_generalization_decreases_and_is_in_mm() -> None:
    rng = np.random.default_rng(2)
    shapes = _two_mode_population(rng)  # ~10 mm scale
    gen = generalization(shapes)
    assert gen[0] > gen[2] > gen[-1]  # adding modes reduces LOO error
    assert gen[-1] < 0.5  # two-mode data is nearly perfectly reconstructed
    assert 0.1 < gen[0] < 100.0  # millimetre-scale, not unitless


def test_specificity_is_reproducible_and_rises_with_modes() -> None:
    rng = np.random.default_rng(3)
    shapes = _two_mode_population(rng)
    modes = np.array([1, 2, 5, 10])
    spec_a = specificity(shapes, modes, n_samples=200, seed=42)
    spec_b = specificity(shapes, modes, n_samples=200, seed=42)
    np.testing.assert_array_equal(spec_a, spec_b)  # seeded -> deterministic
    assert spec_a[-1] > spec_a[0]  # extra modes let samples drift off the data
    assert np.all(spec_a > 0)


def test_evaluate_ssm_bundles_aligned_curves() -> None:
    rng = np.random.default_rng(4)
    shapes = _two_mode_population(rng, n=25, m=15)
    ev = evaluate_ssm(shapes, n_specificity_samples=100, seed=0, label="synthetic")
    assert isinstance(ev, SSMEvaluation)
    assert ev.n_subjects == 25 and ev.n_points == 15
    assert len(ev.modes) == len(ev.compactness) == len(ev.generalization) == len(ev.specificity)
    assert ev.label == "synthetic"
    assert ev.modes_for_variance(0.90) <= ev.modes_for_variance(0.99)
    frame = ev.to_frame()
    assert list(frame.columns) == ["modes", "compactness", "generalization_mm", "specificity_mm"]


def test_default_mode_grid_thins_for_large_models() -> None:
    assert np.array_equal(default_mode_grid(8), np.arange(1, 9))
    big = default_mode_grid(234)
    assert big[0] == 1 and big[-1] == 234
    assert len(big) < 60  # thinned, not every mode
    assert np.all(np.diff(big) > 0)  # strictly ascending, unique


def test_generalization_needs_enough_subjects() -> None:
    with pytest.raises(ValueError, match="at least 3 subjects"):
        generalization(np.zeros((2, 5, 3)))


def test_evaluation_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(S, M, 3\)"):
        compactness(np.zeros((4, 5)))


def test_load_shapeworks_evaluation_reads_studio_csvs(tmp_path: Path) -> None:
    (tmp_path / "compactness.csv").write_text(
        "#  Number of Modes, Explained Variance\n1, 0.6\n2, 0.85\n3, 0.95\n"
    )
    (tmp_path / "generalization.csv").write_text(
        "#  Number of Modes, Generalization\n1, 2.5\n2, 1.8\n3, 1.4\n"
    )
    # specificity.csv intentionally absent -> NaN column
    ev = load_shapeworks_evaluation(tmp_path)
    np.testing.assert_array_equal(ev.modes, [1, 2, 3])
    np.testing.assert_allclose(ev.compactness, [0.6, 0.85, 0.95])
    np.testing.assert_allclose(ev.generalization, [2.5, 1.8, 1.4])
    assert np.all(np.isnan(ev.specificity))
