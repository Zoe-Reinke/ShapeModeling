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

from shape_modeling.analysis import (
    CleanConfig,
    ShapeModel,
    build_ssm,
    centroid_size,
    clean_mask,
    combat_harmonize,
    export_mode_shapes,
    extract_surface,
    gpa,
    hull_surface,
    mode_shape,
    save_ply,
    smooth_surface,
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
from shape_modeling.analysis.surface import Surface
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
