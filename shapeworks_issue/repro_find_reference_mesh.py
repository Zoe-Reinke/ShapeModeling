"""Minimal reproduction: MeshUtils.findReferenceMesh returns a mesh it never compared.

With more than 30 meshes, findReferenceMesh (default random_subset_size=-1 -> 30) only runs
the pairwise ICP on a random subset of 30 meshes, but takes the argmin over a `means` vector
sized for ALL meshes and initialised to 0. Every mesh outside the subset keeps mean 0 and
therefore "wins"; the returned index is the lowest index NOT in the subset.

Test data: 100 concentric spheres with radii 20.0, 19.9, ..., 10.1 mm (index 0 = largest).
After centroid-matched rigid ICP the pairwise distance is ~|r_i - r_j|, so the true medoid is
the median-radius sphere (index ~49/50). The buggy default returns a small index (an extreme
sphere) instead.

Run from the ShapeWorks conda env:  python repro_find_reference_mesh.py
Needs only numpy + scipy (both ship with the ShapeWorks env).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import shapeworks as sw
from scipy.spatial import ConvexHull

N_MESHES = 100
N_POINTS = 400


def write_sphere_ply(path: Path, radius: float, n: int = N_POINTS) -> None:
    """Deterministic Fibonacci-sphere triangulation, faces oriented outward."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    pts = radius * np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    faces = ConvexHull(pts).simplices.copy()
    # orient outward (convex, centred at the origin): normal must point away from the centre
    a, b, c = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    with open(path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
            f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
        )
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def main() -> None:
    radii = 20.0 - 0.1 * np.arange(N_MESHES)  # index 0 = largest (an extreme shape)
    with tempfile.TemporaryDirectory() as tmp:
        meshes = []
        for k, r in enumerate(radii):
            p = Path(tmp) / f"sphere_{k:03d}.ply"
            write_sphere_ply(p, float(r))
            meshes.append(sw.Mesh(str(p)))

        expected = int(np.argsort(radii)[N_MESHES // 2])  # median radius
        default_idx = sw.MeshUtils.findReferenceMesh(meshes)  # what find_reference_image_index uses
        print(f"sw version                : {getattr(sw, '__version__', 'unknown')}")
        print(f"expected medoid (median r): index ~{expected} (r = {radii[expected]:.1f} mm)")
        print(f"findReferenceMesh(default): index {default_idx} (r = {radii[default_idx]:.1f} mm)")

        try:
            full_idx = sw.MeshUtils.findReferenceMesh(meshes, 0)  # 0 = no subset, full pairwise
            print(f"findReferenceMesh(..., 0) : index {full_idx} (r = {radii[full_idx]:.1f} mm)")
        except TypeError:
            full_idx = None
            print("findReferenceMesh(..., 0) : random_subset_size not exposed in this Python binding")

    if abs(radii[default_idx] - radii[expected]) > 1.0:
        print("\n=> BUG REPRODUCED: the default call returned an extreme mesh, not the medoid.")
    else:
        print("\n=> Not reproduced: the default call returned (close to) the medoid.")


if __name__ == "__main__":
    main()
