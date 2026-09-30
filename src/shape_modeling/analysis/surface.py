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

import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree
from skimage import measure

from shape_modeling.analysis.ssm import ShapeModel, mode_shape
from shape_modeling.io.reader import LoadedVolume

__all__ = [
    "CleanConfig",
    "MeshQC",
    "Surface",
    "clean_mask",
    "export_mode_shapes",
    "extract_surface",
    "hull_surface",
    "qc_mesh",
    "reconstruct_surface",
    "save_ply",
    "self_intersecting_pairs",
    "smooth_surface",
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


def _tet_circumspheres(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Circumsphere centre and radius of each tetrahedron (``nan``/``inf`` if degenerate)."""
    origin = points[tets[:, 0]]
    rows = points[tets[:, 1:]] - origin[:, None, :]  # (T, 3, 3)
    rhs = 0.5 * np.einsum("tij,tij->ti", rows, rows)
    centres = np.full((len(tets), 3), np.nan)
    radii = np.full(len(tets), np.inf)
    ok = np.abs(np.linalg.det(rows)) > 1e-12
    if ok.any():
        local = np.linalg.solve(rows[ok], rhs[ok][..., None])[..., 0]
        centres[ok] = local + origin[ok]
        radii[ok] = np.linalg.norm(local, axis=1)
    return centres, radii


def reconstruct_surface(points: np.ndarray, *, alpha: float | None = None) -> Surface:
    """Reconstruct a closed, concavity-following surface **on** a corresponded point cloud.

    Unlike :func:`hull_surface` this is a real reconstruction: it recovers concavities (a
    thyroid lobe's medial face, a prostate's base notch) instead of bridging them. Unlike a
    Poisson / ball-pivoting reconstruction it introduces **no new vertices** -- the returned
    ``faces`` index ``points`` in their original order, so the same face list stays valid for
    every shape the model can generate (``x = mean + sum_k b_k sqrt(lambda_k) phi_k`` moves
    vertices, never topology) and can be handed to a differentiable renderer/SDF as-is.

    Method (Delaunay *sculpting*): tetrahedralise ``points``, then repeatedly carve the
    largest-circumsphere tetrahedron off the solid's boundary, accepting a removal only when
    the boundary stays a closed 2-manifold and no point is pushed back into the interior. The
    result is therefore watertight, edge/vertex-manifold, consistently outward-oriented and
    free of self-intersections **by construction** (it bounds a union of tetrahedra), and uses
    all ``M`` points -- giving exactly ``2M - 4`` triangles for a genus-0 organ.

    What stops the carving is a tetrahedron's **circumcentre**, not a radius threshold: a
    tetrahedron bridging a concavity has its circumcentre out in the exterior (the Voronoi-pole
    property), while one filling the organ's interior has it near the medial axis, inside. So
    carve while the circumcentre is outside the solid and stop when it is inside -- which needs
    no parameter and adapts to the organ's thickness. A plain alpha-radius rule cannot: sampled
    on its surface only, a solid organ's interior is spanned by tetrahedra of circumradius
    ~half its thickness, so an ``alpha`` small enough to cut into a real concavity also hollows
    the organ out (a two-sphere dumbbell loses 46% of its volume; a thyroid lobe at ``alpha =
    1.5x`` the point spacing loses 55%).

    ``alpha`` (mm) optionally carves *deeper* still, removing any boundary tetrahedron above
    that circumradius even when its circumcentre is interior; the default ``None`` leaves the
    criterion above alone, and small values hollow the shape out as described.

    Validated on thyroid lobes against the groomed distance transforms their particles were
    optimised on: 0.25 mm mean symmetric surface error and -2% volume, where
    :func:`hull_surface` on the same particles gives 0.24-1.2 mm and up to +58% (it is exact
    only where a lobe happens to be convex).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (M, 3), got {pts.shape}")
    if len(pts) < 4:
        raise ValueError(f"need at least 4 points to reconstruct a surface, got {len(pts)}")
    carve_above = np.inf if alpha is None else float(alpha)

    try:
        triangulation = Delaunay(pts)
    except QhullError as exc:  # collapsed / coplanar / duplicated particles
        raise ValueError(
            f"cannot tetrahedralise this point cloud (extent {np.ptp(pts, axis=0).round(3)}); "
            "it is degenerate -- collapsed, coplanar or duplicated points"
        ) from exc
    tets = triangulation.simplices
    centres, radii = _tet_circumspheres(pts, tets)
    # Which tetrahedron holds each circumcentre. The triangulation never changes, so this
    # lookup is static -- only whether that tetrahedron is still *inside* changes as we carve.
    home = triangulation.find_simplex(np.nan_to_num(centres, nan=np.inf))

    # Face bookkeeping: every triangle of every tetrahedron, and how many *inside* tetrahedra
    # it belongs to. A face on the solid's boundary belongs to exactly one.
    face_id: dict[tuple[int, ...], int] = {}
    tet_faces = np.empty((len(tets), 4), dtype=np.int64)
    for ti, tet in enumerate(tets):
        for k in range(4):
            key = tuple(sorted(int(v) for v in np.delete(tet, k)))
            tet_faces[ti, k] = face_id.setdefault(key, len(face_id))
    faces_all = np.array(sorted(face_id, key=face_id.__getitem__), dtype=np.int64)
    n_inside = np.zeros(len(faces_all), dtype=np.int32)
    np.add.at(n_inside, tet_faces.ravel(), 1)
    face_tets: list[list[int]] = [[] for _ in range(len(faces_all))]
    for ti in range(len(tets)):
        for fi in tet_faces[ti]:
            face_tets[fi].append(ti)

    # Boundary-incidence counters, kept in sync as faces enter/leave the boundary, so the
    # manifold test for a candidate removal is O(1).
    edge_hits: dict[tuple[int, ...], int] = {}
    vert_hits = np.zeros(len(pts), dtype=np.int32)

    def touch(fi: int, delta: int) -> None:
        a, b, c = faces_all[fi]
        for edge in ((a, b), (a, c), (b, c)):
            edge_hits[edge] = edge_hits.get(edge, 0) + delta
        vert_hits[faces_all[fi]] += delta

    for fi in np.flatnonzero(n_inside == 1):
        touch(int(fi), +1)

    inside = np.ones(len(tets), dtype=bool)
    heap = [(-radii[ti], int(ti)) for ti in np.flatnonzero((n_inside[tet_faces] == 1).any(axis=1))]
    heapq.heapify(heap)

    def removable(ti: int) -> bool:
        """Can this tetrahedron be carved off without tearing or pinching the boundary?

        One boundary face: the opposite vertex is newly exposed, so it must not already be on
        the boundary (that would pinch two sheets together at a point). Two boundary faces:
        the edge opposite their shared one is newly exposed and must likewise be free. Three or
        four boundary faces would *remove* a vertex from the surface -- refused, because every
        correspondence point has to stay on the mesh.
        """
        boundary = [int(fi) for fi in tet_faces[ti] if n_inside[fi] == 1]
        if len(boundary) == 1:
            opposite = set(int(v) for v in tets[ti]) - set(int(v) for v in faces_all[boundary[0]])
            return vert_hits[opposite.pop()] == 0
        if len(boundary) == 2:
            shared = set(faces_all[boundary[0]]) & set(faces_all[boundary[1]])
            rest = tuple(sorted(int(v) for v in set(int(v) for v in tets[ti]) - shared))
            return edge_hits.get(rest, 0) == 0
        return False

    def carve(ti: int) -> None:
        inside[ti] = False
        for fi in tet_faces[ti]:
            if n_inside[fi] == 1:  # was boundary, now gone
                touch(int(fi), -1)
                n_inside[fi] = 0
            else:  # was interior, now boundary
                n_inside[fi] = 1
                touch(int(fi), +1)
                for neighbour in face_tets[fi]:
                    if inside[neighbour]:
                        heapq.heappush(heap, (-radii[neighbour], neighbour))

    while heap:
        neg_radius, ti = heapq.heappop(heap)
        if not inside[ti] or not removable(ti):
            continue
        circumcentre_outside = home[ti] < 0 or not inside[home[ti]]
        if not circumcentre_outside and -neg_radius <= carve_above:
            continue
        carve(ti)

    # A point can still end up buried where the surface is so pinched that the circumcentre
    # rule stops before reaching it. Dig those out one at a time, touching only tetrahedra that
    # actually contain a buried point: carving indiscriminately until everything is exposed is
    # what the (rejected) radius rule does, and it eats the shape (see the docstring).
    while (buried := np.flatnonzero(vert_hits == 0)).size:
        wanted = np.isin(tets, buried).any(axis=1) & inside
        candidates = [
            (-radii[ti], int(ti))
            for ti in np.flatnonzero(wanted & (n_inside[tet_faces] == 1).any(axis=1))
        ]
        for _, ti in sorted(candidates):  # largest circumsphere first, as above
            if removable(ti):
                carve(ti)
                break
        else:
            raise ValueError(
                f"{buried.size} of {len(pts)} points could not be exposed on the surface; the "
                "point cloud is probably degenerate (collapsed or duplicated particles)"
            )

    # Orient every boundary triangle away from the tetrahedron it still bounds -> outward
    # normals, consistently wound, with no BFS needed.
    boundary_faces = np.flatnonzero(n_inside == 1)
    out = np.empty((len(boundary_faces), 3), dtype=np.int64)
    for row, fi in enumerate(boundary_faces):
        tri = faces_all[fi]
        ti = next(t for t in face_tets[fi] if inside[t])
        apex = (set(int(v) for v in tets[ti]) - set(int(v) for v in tri)).pop()
        a, b, c = pts[tri]
        if np.dot(np.cross(b - a, c - a), pts[apex] - a) > 0:
            tri = tri[::-1]
        out[row] = tri
    return Surface(verts=pts, faces=out)


def _orientable(faces: np.ndarray) -> bool:
    """Can the triangles be re-wound into a consistent orientation? (BFS over shared edges.)"""
    sides: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(min(u, v)), int(max(u, v)))
            sides.setdefault(key, []).append((fi, u < v))
    if any(len(s) > 2 for s in sides.values()):
        return False  # non-manifold edge: orientation is not well defined
    flip = np.full(len(faces), -1, dtype=np.int8)
    for start in range(len(faces)):
        if flip[start] != -1:
            continue
        flip[start] = 0
        stack = [start]
        while stack:
            fi = stack.pop()
            a, b, c = faces[fi]
            for u, v in ((a, b), (b, c), (c, a)):
                key = (int(min(u, v)), int(max(u, v)))
                for gj, forward in sides[key]:
                    if gj == fi:
                        continue
                    # Same traversal direction => the two faces must disagree on the flip.
                    want = flip[fi] ^ (1 if forward == (u < v) else 0)
                    if flip[gj] == -1:
                        flip[gj] = want
                        stack.append(gj)
                    elif flip[gj] != want:
                        return False
    return True


@dataclass
class MeshQC:
    """Topological/geometric health of a triangle mesh (see :func:`qc_mesh`)."""

    n_vertices: int
    n_faces: int
    n_unreferenced_vertices: int  # points no triangle uses -- dead weight in a shape model
    n_non_manifold_edges: int  # edges not shared by exactly two triangles
    n_self_intersections: int  # intersecting triangle pairs that share no vertex
    is_edge_manifold: bool
    is_vertex_manifold: bool
    is_orientable: bool  # a consistent winding exists
    is_consistently_oriented: bool  # the winding it *has* is consistent and outward
    is_self_intersecting: bool
    is_watertight: bool

    @property
    def ok(self) -> bool:
        """True when the mesh is safe to hand to a renderer / SDF / volume computation."""
        return (
            self.is_watertight
            and self.is_consistently_oriented
            and not self.is_self_intersecting
            and self.n_unreferenced_vertices == 0
        )

    def summary(self) -> str:
        """One line, e.g. ``ok  V=512 F=1020 ...`` -- for logging a per-instance sweep."""
        flags = "ok " if self.ok else "BAD"
        return (
            f"{flags} V={self.n_vertices} F={self.n_faces} "
            f"unused={self.n_unreferenced_vertices} nonmanifold_edges={self.n_non_manifold_edges} "
            f"self_intersections={self.n_self_intersections} watertight={self.is_watertight} "
            f"oriented={self.is_consistently_oriented}"
        )


def _segment_triangle_hits(
    seg_a: np.ndarray, seg_b: np.ndarray, tri: np.ndarray, *, eps: float = 1e-9
) -> np.ndarray:
    """Moeller-Trumbore: does each segment cross the interior of its paired triangle?"""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2, d = b - a, c - a, seg_b - seg_a
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > eps
    inv = np.zeros(len(det))
    inv[ok] = 1.0 / det[ok]
    tvec = seg_a - a
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,ij->i", d, qvec) * inv
    t = np.einsum("ij,ij->i", e2, qvec) * inv
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def self_intersecting_pairs(surface: Surface) -> np.ndarray:
    """The ``(P, 2)`` pairs of triangle indices that cross each other, sharing no vertex.

    Use it to *see* where a mesh folds -- ``qc_mesh`` only counts these. A fixed face list
    reused across a shape model's instances is the usual source: the topology is clean on the
    mean and folds where the modes move vertices past each other (see :func:`qc_mesh`).

    Broad phase: overlapping axis-aligned bounding boxes, from a KD-tree on face centroids.
    Narrow phase: a triangle pair intersects iff an edge of one pierces the other (the only
    case this misses is two exactly-coplanar overlapping triangles, which a deformed
    correspondence mesh does not produce).
    """
    verts = np.asarray(surface.verts, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    tri = verts[faces]
    centroids = tri.mean(axis=1)
    reach = float(np.linalg.norm(tri - centroids[:, None, :], axis=2).max())
    pairs = np.array(sorted(cKDTree(centroids).query_pairs(2 * reach)), dtype=np.int64)
    empty = np.empty((0, 2), dtype=np.int64)
    if not len(pairs):
        return empty
    i, j = pairs[:, 0], pairs[:, 1]
    shares_vertex = (faces[i][:, :, None] == faces[j][:, None, :]).any(axis=(1, 2))
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    overlaps = ((lo[i] <= hi[j]) & (lo[j] <= hi[i])).all(axis=1)
    i, j = i[~shares_vertex & overlaps], j[~shares_vertex & overlaps]
    if not len(i):
        return empty
    hit = np.zeros(len(i), dtype=bool)
    for src, dst in ((i, j), (j, i)):
        for k in range(3):
            hit |= _segment_triangle_hits(
                verts[faces[src][:, k]], verts[faces[src][:, (k + 1) % 3]], tri[dst]
            )
    return np.stack([i[hit], j[hit]], axis=1)


def _link_is_one_cycle(link: list[tuple[int, int]]) -> bool:
    """Is a vertex's link (the face edges opposite it) a single closed loop?"""
    degree: dict[int, int] = {}
    adjacency: dict[int, list[int]] = {}
    for a, b in link:
        for x, y in ((a, b), (b, a)):
            degree[x] = degree.get(x, 0) + 1
            adjacency.setdefault(x, []).append(y)
    if not degree or any(d != 2 for d in degree.values()):
        return False
    seen = {next(iter(degree))}
    stack = [next(iter(degree))]
    while stack:
        for nb in adjacency[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(degree)


def qc_mesh(surface: Surface) -> MeshQC:
    """Manifold / watertight / self-intersection / orientation checks on a triangle mesh.

    Ported from the Open3D-based ``qc_mesh`` in the downstream ShapeModelReconstruction
    project (``mesh.py``) so the SSM side can gate an export without that dependency --
    ``is_watertight`` follows Open3D's definition (edge- and vertex-manifold, closed, not
    self-intersecting). Run it on the **mean and on sampled/fitted instances**: a mesh whose
    topology is fine on the mean can still fold once the modes move its vertices, and a convex
    hull passes on the mean for free.
    """
    verts = np.asarray(surface.verts, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    n_non_manifold = int(np.count_nonzero(counts != 2))

    links: dict[int, list[tuple[int, int]]] = {}
    for a, b, c in faces:
        links.setdefault(int(a), []).append((int(b), int(c)))
        links.setdefault(int(b), []).append((int(c), int(a)))
        links.setdefault(int(c), []).append((int(a), int(b)))
    vertex_manifold = all(map(_link_is_one_cycle, links.values()))  # over *used* vertices

    # A consistent winding exists iff no two triangles traverse a shared edge the same way
    # (checked on the winding as given; that also tells us whether it is *already* consistent).
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, dup_counts = np.unique(directed, axis=0, return_counts=True)
    consistent_winding = bool((dup_counts == 1).all())
    signed_volume = float(
        np.einsum("ij,ij->i", verts[faces[:, 0]], np.cross(verts[faces[:, 1]], verts[faces[:, 2]]))
        .sum()
        / 6.0
    )
    n_hits = len(self_intersecting_pairs(Surface(verts=verts, faces=faces)))
    edge_manifold = n_non_manifold == 0
    return MeshQC(
        n_vertices=len(verts),
        n_faces=len(faces),
        n_unreferenced_vertices=len(verts) - len(np.unique(faces)),
        n_non_manifold_edges=n_non_manifold,
        n_self_intersections=n_hits,
        is_edge_manifold=edge_manifold,
        is_vertex_manifold=bool(vertex_manifold),
        is_orientable=bool(consistent_winding or _orientable(faces)),
        is_consistently_oriented=bool(consistent_winding and signed_volume > 0),
        is_self_intersecting=n_hits > 0,
        is_watertight=bool(edge_manifold and vertex_manifold and n_hits == 0),
    )


def _similarity_fit(
    src: np.ndarray, dst: np.ndarray, *, allow_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (uniform scale ``s``, rotation ``R``, translation ``t``).

    Umeyama (1991): the best ``s, R, t`` with ``dst ≈ s * (src @ R.T) + t``. Used to absorb the
    bulk frame/scale change before a thin-plate spline, so the spline only models the small
    residual deformation (a raw spline forced to do the scale change too oscillates and folds).

    ``allow_scale=False`` pins ``s = 1`` and returns the best **rigid** (6-DOF) fit instead --
    the right solve for a model that already lives in millimetres (``gpa(scale=False)``), where
    letting the pose rescale would throw away exactly the size the model was built to keep.
    """
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    corr = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # keep a proper (det=+1) rotation
        corr[-1, -1] = -1.0
    rot = u @ corr @ vt
    if allow_scale:
        var_src = (src_c**2).sum() / src.shape[0]
        scale = float((d * np.diag(corr)).sum() / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0
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


def smooth_surface(
    surface: Surface,
    *,
    iterations: int = 15,
    lam: float = 0.5,
    mu: float = -0.53,
) -> Surface:
    """Taubin (λ|μ) smoothing of a triangle mesh; faces unchanged, gross shape preserved.

    Removes high-frequency vertex noise -- marching-cubes staircase faceting, or the surface
    texture a single real gland injects when its mesh is warped onto an *average* shape (see
    :func:`warp_surface`). Plain Laplacian smoothing would also shrink the mesh toward its
    centroid; Taubin alternates a positive Laplacian step (``lam``, smooth + shrink) with a
    negative one (``mu`` slightly larger in magnitude, smooth + re-inflate) so volume is largely
    retained over the ``iterations`` passes.

    Uses a uniform-weight graph Laplacian built from the triangle edges (no geometry weighting),
    which is enough to strip sub-voxel bumps from an organ surface. Returns a new :class:`Surface`.
    """
    verts = np.asarray(surface.verts, dtype=np.float64)
    faces = np.asarray(surface.faces)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) verts, got {verts.shape}")
    n = len(verts)

    # Undirected vertex adjacency from triangle edges (each edge added both ways, then de-duped).
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    adj = sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0  # collapse duplicate edges to weight 1

    deg = np.asarray(adj.sum(axis=1)).ravel()
    # Uniform-weight Laplacian L = D^-1 A - I (each row averages neighbours and subtracts self).
    lap = sparse.diags(1.0 / np.maximum(deg, 1.0)) @ adj - sparse.eye(n)

    x = verts.copy()
    for k in range(iterations):
        x = x + (lam if k % 2 == 0 else mu) * (lap @ x)
    return Surface(verts=x, faces=surface.faces)


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
