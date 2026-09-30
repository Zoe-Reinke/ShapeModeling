"""Tests for the numpy-only parts of shape_modeling.analysis.reference (no ShapeWorks needed)."""

from __future__ import annotations

import numpy as np
import pytest

from shape_modeling.analysis.reference import (
    chunk_bounds,
    means_from_pairs,
    n_pairs,
    pair_from_flat,
)


def _shapeworks_pairs(n: int) -> list[tuple[int, int]]:
    """The enumeration order of MeshUtils::findReferenceMesh."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


@pytest.mark.parametrize("n", [2, 3, 7, 31, 100])
def test_pair_from_flat_matches_shapeworks_order(n: int) -> None:
    i, j = pair_from_flat(np.arange(n_pairs(n)), n)
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == _shapeworks_pairs(n)


def test_pair_from_flat_large_n_endpoints() -> None:
    n = 1054
    i, j = pair_from_flat(np.array([0, 1, n - 2, n - 1, n_pairs(n) - 1]), n)
    assert list(zip(i.tolist(), j.tolist(), strict=True)) == [
        (0, 1), (0, 2), (0, n - 1), (1, 2), (n - 2, n - 1)
    ]
    # every row boundary maps correctly
    starts = np.array([r * n - r * (r + 1) // 2 for r in range(n - 1)])
    ri, rj = pair_from_flat(starts, n)
    assert np.array_equal(ri, np.arange(n - 1))
    assert np.array_equal(rj, np.arange(1, n))


@pytest.mark.parametrize("n,n_chunks", [(10, 1), (10, 7), (1054, 200), (1054, 1000)])
def test_chunks_partition_all_pairs(n: int, n_chunks: int) -> None:
    bounds = [chunk_bounds(n, c, n_chunks) for c in range(n_chunks)]
    assert bounds[0][0] == 0 and bounds[-1][1] == n_pairs(n)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
    sizes = [hi - lo for lo, hi in bounds]
    assert max(sizes) - min(sizes) <= 1


def test_chunk_out_of_range() -> None:
    with pytest.raises(ValueError):
        chunk_bounds(10, 5, 5)


def test_means_from_pairs_picks_median_for_line_metric() -> None:
    # d(i, j) = |x_i - x_j|: the medoid of points on a line is the median point
    rng = np.random.default_rng(0)
    x = rng.permutation(np.linspace(0.0, 10.0, 41))
    n = len(x)
    i, j = pair_from_flat(np.arange(n_pairs(n)), n)
    means = means_from_pairs(n, i, j, np.abs(x[i] - x[j]))
    assert int(np.argmin(means)) == int(np.argsort(x)[n // 2])
    # normalisation: sum over the other n-1 shapes / (n - 1)
    k = 3
    assert np.isclose(means[k], np.abs(x - x[k]).sum() / (n - 1))
