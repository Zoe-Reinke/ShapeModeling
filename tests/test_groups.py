"""Tests for shape_modeling.analysis.groups (healthy-vs-cancer group analysis).

Pure synthetic score arrays -- no ShapeWorks, no real data. The key cases are a *positive
control* (a planted group difference must be detected) and a *null control* (identical groups
must not produce a false positive).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from shape_modeling.analysis.cohort import (
    CohortEntry,
    build_cohort,
    build_grouped_cohort,
)
from shape_modeling.analysis.groups import (
    compare_groups,
    group_mean_difference_shape,
    per_mode_tests,
    shape_predicts_cancer,
    volume_difference,
)
from shape_modeling.analysis.ssm import ShapeModel
from shape_modeling.io.reader import Case, Dataset, LoadedVolume, VolumeType

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mask_volume(volume_type: VolumeType = VolumeType.ORGAN_SEG) -> LoadedVolume:
    """A tiny non-empty mask wrapped as a LoadedVolume (1 mm isotropic)."""
    array = np.zeros((6, 6, 6), np.uint8)
    array[2:4, 2:4, 2:4] = 1
    img = sitk.GetImageFromArray(array)
    img.SetSpacing((1.0, 1.0, 1.0))
    return LoadedVolume(
        path=Path("synthetic.nrrd"),
        volume_type=volume_type,
        size=tuple(img.GetSize()),
        spacing=(1.0, 1.0, 1.0),
        origin=img.GetOrigin(),
        direction=img.GetDirection(),
        dtype="uint8",
        sitk_image=img,
        array=array,
    )


def _case(case_id: str, *, organ: bool = True, cancer: bool = False) -> Case:
    return Case(
        case_id=case_id,
        organ_segs=[_mask_volume()] if organ else [],
        cancer_segs=[_mask_volume(VolumeType.CANCER_SEG)] if cancer else [],
    )


def _grouped_scores(
    n_per_group: int, n_modes: int, offset: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Scores for ``2*n_per_group`` subjects; cancer group shifted on mode 0 by ``offset``."""
    rng = np.random.default_rng(seed)
    healthy = rng.normal(size=(n_per_group, n_modes))
    cancer = rng.normal(size=(n_per_group, n_modes))
    cancer[:, 0] += offset
    scores = np.vstack([healthy, cancer])
    is_cancer = np.array([False] * n_per_group + [True] * n_per_group)
    return scores, is_cancer


# --------------------------------------------------------------------------- #
# build_grouped_cohort
# --------------------------------------------------------------------------- #


def test_build_grouped_cohort_flags_and_keeps_both_groups() -> None:
    ds = Dataset(
        cases=[
            _case("healthy0"),
            _case("healthy1"),
            _case("cancer0", cancer=True),
            _case("no_organ", organ=False, cancer=True),  # skipped: no gland to model
        ]
    )

    grouped = build_grouped_cohort({"Synthetic": ds})

    assert len(grouped) == 3  # the organ-less case is dropped
    by_id = {e.case_id: e for e in grouped}
    assert by_id["cancer0"].is_cancer is True
    assert by_id["healthy0"].is_cancer is False
    assert by_id["healthy1"].is_cancer is False
    # Sorted by (source, is_cancer, case_id): healthy cases come before cancer cases.
    assert [e.case_id for e in grouped] == ["healthy0", "healthy1", "cancer0"]


def test_build_cohort_still_excludes_cancer() -> None:
    """The healthy-only builder must be unchanged by the new is_cancer field."""
    ds = Dataset(cases=[_case("h0"), _case("c0", cancer=True)])
    healthy = build_cohort({"Synthetic": ds})
    assert [e.case_id for e in healthy] == ["h0"]
    assert all(e.is_cancer is False for e in healthy)


# --------------------------------------------------------------------------- #
# per_mode_tests
# --------------------------------------------------------------------------- #


def test_per_mode_tests_positive_control() -> None:
    scores, is_cancer = _grouped_scores(n_per_group=40, n_modes=5, offset=2.0, seed=0)
    tests = per_mode_tests(scores, is_cancer, top_k=5)

    assert len(tests) == 5
    mode0 = tests[0]
    assert mode0.p_adjusted < 0.01  # the planted mode is flagged after FDR correction
    assert mode0.cohens_d > 1.0  # large positive effect (cancer scores higher)
    assert mode0.mean_cancer > mode0.mean_healthy
    # The other modes carry no signal and should not be flagged.
    assert all(t.p_adjusted > 0.05 for t in tests[1:])


def test_per_mode_tests_null_control() -> None:
    scores, is_cancer = _grouped_scores(n_per_group=40, n_modes=5, offset=0.0, seed=1)
    tests = per_mode_tests(scores, is_cancer, top_k=5)
    assert all(t.p_adjusted > 0.05 for t in tests)  # no false positives


def test_per_mode_tests_top_k_limits_family() -> None:
    scores, is_cancer = _grouped_scores(n_per_group=20, n_modes=8, offset=1.0, seed=2)
    assert len(per_mode_tests(scores, is_cancer, top_k=3)) == 3
    assert len(per_mode_tests(scores, is_cancer)) == 8  # default: all modes


# --------------------------------------------------------------------------- #
# shape_predicts_cancer (omnibus)
# --------------------------------------------------------------------------- #


def test_shape_predicts_cancer_positive_control() -> None:
    scores, is_cancer = _grouped_scores(n_per_group=40, n_modes=5, offset=2.0, seed=3)
    res = shape_predicts_cancer(scores, is_cancer, top_k=5, n_permutations=199, seed=0)
    assert res.auc > 0.8  # strongly separable
    assert res.p_value < 0.05  # permutation-significant
    assert res.top_k == 5


def test_shape_predicts_cancer_null_control() -> None:
    scores, is_cancer = _grouped_scores(n_per_group=40, n_modes=5, offset=0.0, seed=4)
    res = shape_predicts_cancer(scores, is_cancer, top_k=5, n_permutations=199, seed=0)
    assert 0.3 < res.auc < 0.7  # near chance
    assert res.p_value > 0.05  # not significant


# --------------------------------------------------------------------------- #
# group_mean_difference_shape
# --------------------------------------------------------------------------- #


def test_group_mean_difference_recovers_planted_offset() -> None:
    rng = np.random.default_rng(5)
    n_points, n_modes = 6, 2
    mean = rng.normal(size=(n_points, 3))
    components = rng.normal(size=(n_modes, n_points * 3))

    healthy_score = np.array([0.5, -0.3])
    cancer_score = np.array([1.5, 0.7])
    scores = np.vstack([np.tile(healthy_score, (10, 1)), np.tile(cancer_score, (10, 1))])
    is_cancer = np.array([False] * 10 + [True] * 10)

    model = ShapeModel(
        mean=mean,
        components=components,
        eigenvalues=np.array([1.0, 0.5]),
        explained_variance_ratio=np.array([0.67, 0.33]),
        scores=scores,
    )

    mean_h, mean_c, delta = group_mean_difference_shape(model, is_cancer)
    expected_delta = ((cancer_score - healthy_score) @ components).reshape(n_points, 3)
    assert np.allclose(delta, expected_delta)
    assert np.allclose(mean_c - mean_h, delta)


# --------------------------------------------------------------------------- #
# volume_difference
# --------------------------------------------------------------------------- #


def test_volume_difference_detects_size_gap() -> None:
    organ = _mask_volume()
    cohort = (
        [CohortEntry(f"h{i}", "S", 40.0 + i, organ, is_cancer=False) for i in range(15)]
        + [CohortEntry(f"c{i}", "S", 80.0 + i, organ, is_cancer=True) for i in range(15)]
    )
    vd = volume_difference(cohort)
    assert vd.n_healthy == 15
    assert vd.n_cancer == 15
    assert vd.median_cancer > vd.median_healthy
    assert vd.p_value < 0.01
    assert vd.cohens_d > 1.0


# --------------------------------------------------------------------------- #
# compare_groups + guards
# --------------------------------------------------------------------------- #


def test_compare_groups_bundles_results() -> None:
    rng = np.random.default_rng(6)
    n_points, n_modes = 5, 5
    scores, is_cancer = _grouped_scores(n_per_group=30, n_modes=n_modes, offset=2.0, seed=6)
    model = ShapeModel(
        mean=rng.normal(size=(n_points, 3)),
        components=rng.normal(size=(n_modes, n_points * 3)),
        eigenvalues=np.linspace(5, 1, n_modes),
        explained_variance_ratio=np.full(n_modes, 0.2),
        scores=scores,
    )
    result = compare_groups(model, is_cancer, top_k=5, n_permutations=99, seed=0)
    assert result.n_healthy == 30
    assert result.n_cancer == 30
    assert result.significant_modes()  # at least the planted mode survives FDR
    assert result.omnibus.auc > 0.8
    assert result.mean_delta.shape == (n_points, 3)
    assert result.volume is None  # no cohort passed


def test_group_helpers_require_both_groups() -> None:
    scores = np.random.default_rng(7).normal(size=(10, 4))
    all_healthy = np.zeros(10, dtype=bool)
    with pytest.raises(ValueError, match="both cancer and healthy"):
        per_mode_tests(scores, all_healthy)
    with pytest.raises(ValueError, match="both cancer and healthy"):
        shape_predicts_cancer(scores, all_healthy)
