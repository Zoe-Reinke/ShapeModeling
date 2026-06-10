"""Healthy-vs-cancer group analysis over the shape model (pipeline stage 6).

The exploratory SSM (:mod:`~shape_modeling.analysis.ssm`) gives every subject a vector of
mode *scores*. This module asks the project's headline question on top of those scores:
**does the whole-gland shape differ between glands that harbour clinically significant cancer
and benign ones?** A cancer case's lesion mask only sets its group label (via
:func:`~shape_modeling.analysis.cohort.build_grouped_cohort`); the geometry compared is the
gland itself.

Everything here operates on plain arrays -- the ``(S, K)`` score matrix from
:class:`~shape_modeling.analysis.ssm.ShapeModel` and a boolean group vector aligned to the
same subject order -- so it is independent of ShapeWorks and unit-testable on synthetic
scores. Two complementary readings:

* :func:`per_mode_tests` -- *which* modes differ: a per-mode Mann-Whitney U test plus Cohen's
  d effect size, with Benjamini-Hochberg FDR correction across the tested modes.
* :func:`shape_predicts_cancer` -- an omnibus *does shape predict cancer at all?* test:
  cross-validated AUC of a logistic model on the mode scores, with a label-permutation
  p-value (chance AUC = 0.5).

:func:`group_mean_difference_shape` reconstructs each group's mean gland for visualising
*where* the groups differ, and :func:`volume_difference` reports the raw size difference
separately (size is *not* removed when the model is built with ``gpa(scale=False)``).

**Caveat:** correspondence in this pipeline is exploratory ("good enough to look at"), so
treat any group effect as suggestive rather than a validated biomarker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shape_modeling.analysis.cohort import CohortEntry
from shape_modeling.analysis.ssm import ShapeModel

__all__ = [
    "GroupComparison",
    "ModeTest",
    "OmnibusResult",
    "VolumeDifference",
    "compare_groups",
    "group_mean_difference_shape",
    "per_mode_tests",
    "shape_predicts_cancer",
    "volume_difference",
]


def _as_bool_groups(is_cancer: np.ndarray, n: int) -> np.ndarray:
    """Validate the group vector: 1-D, length ``n``, and both groups present."""
    groups = np.asarray(is_cancer, dtype=bool)
    if groups.shape != (n,):
        raise ValueError(f"is_cancer must have shape ({n},), got {groups.shape}")
    if groups.all() or (~groups).all():
        raise ValueError("need both cancer and healthy cases to compare groups")
    return groups


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for ``a`` vs ``b`` (positive when ``a`` has the larger mean)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


@dataclass
class ModeTest:
    """Per-mode healthy-vs-cancer comparison of PCA scores."""

    mode: int  # 0-based mode index
    u_statistic: float  # Mann-Whitney U
    p_value: float  # raw two-sided p
    p_adjusted: float  # Benjamini-Hochberg FDR-adjusted p across the tested modes
    cohens_d: float  # effect size, cancer minus healthy (positive => cancer scores higher)
    mean_healthy: float
    mean_cancer: float


def per_mode_tests(
    scores: np.ndarray,
    is_cancer: np.ndarray,
    *,
    top_k: int | None = None,
) -> list[ModeTest]:
    """Per-mode Mann-Whitney U + Cohen's d, FDR-corrected across the tested modes.

    Args:
        scores: ``(S, K)`` PCA score matrix from :class:`ShapeModel` (rows are subjects in the
            same order as the group vector).
        is_cancer: boolean ``(S,)`` group labels (``True`` = cancer).
        top_k: test only the first ``top_k`` modes. PCA modes are variance-ordered, so the
            leading modes carry the shape signal; restricting the family keeps FDR power up.
            Defaults to all modes -- pass a modest value (e.g. 10) on a high-dimensional model.

    Returns:
        One :class:`ModeTest` per tested mode, with Benjamini-Hochberg adjusted p-values.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be (S, K), got {scores.shape}")
    groups = _as_bool_groups(is_cancer, scores.shape[0])

    k = scores.shape[1] if top_k is None else min(top_k, scores.shape[1])
    cancer, healthy = scores[groups], scores[~groups]

    raw_p, stats_u, ds = [], [], []
    for m in range(k):
        u, p = stats.mannwhitneyu(cancer[:, m], healthy[:, m], alternative="two-sided")
        raw_p.append(p)
        stats_u.append(float(u))
        ds.append(_cohens_d(cancer[:, m], healthy[:, m]))

    # Benjamini-Hochberg FDR control across the tested family (scipy >= 1.11).
    adjusted = stats.false_discovery_control(raw_p, method="bh") if raw_p else np.array([])

    return [
        ModeTest(
            mode=m,
            u_statistic=stats_u[m],
            p_value=float(raw_p[m]),
            p_adjusted=float(adjusted[m]),
            cohens_d=ds[m],
            mean_healthy=float(healthy[:, m].mean()),
            mean_cancer=float(cancer[:, m].mean()),
        )
        for m in range(k)
    ]


@dataclass
class OmnibusResult:
    """Result of the 'does shape predict cancer at all?' permutation test."""

    auc: float  # cross-validated ROC AUC on the real labels (chance = 0.5)
    p_value: float  # permutation p: P(AUC_permuted >= AUC_real)
    n_permutations: int
    top_k: int  # number of leading modes used as features
    n_splits: int


def _cv_auc(
    features: np.ndarray, labels: np.ndarray, n_splits: int, seed: int
) -> float:
    """Out-of-fold ROC AUC of a standardised logistic model (stratified CV)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    oof = cross_val_predict(model, features, labels, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, oof))


def shape_predicts_cancer(
    scores: np.ndarray,
    is_cancer: np.ndarray,
    *,
    top_k: int = 10,
    n_splits: int = 5,
    n_permutations: int = 1000,
    seed: int = 0,
) -> OmnibusResult:
    """Omnibus test: can a logistic model on the mode scores separate the groups?

    Cross-validated ROC AUC of L2 logistic regression on the first ``top_k`` (variance-ordered)
    mode scores, with a label-permutation null. The permutation p-value is the fraction of
    label shuffles whose CV AUC matches or beats the real one -- robust where a parametric
    multivariate test (e.g. Hotelling's T^2) would be unstable at ``S`` ~ ``K``.

    Returns:
        An :class:`OmnibusResult`; ``auc`` near 0.5 with a non-significant ``p_value`` means
        gland shape does not distinguish the groups in this cohort.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be (S, K), got {scores.shape}")
    groups = _as_bool_groups(is_cancer, scores.shape[0])

    k = min(top_k, scores.shape[1])
    features = scores[:, :k]
    labels = groups.astype(int)

    observed = _cv_auc(features, labels, n_splits, seed)

    rng = np.random.default_rng(seed)
    ge = 1  # +1 (the observed statistic) for an unbiased permutation p-value
    for _ in range(n_permutations):
        permuted = rng.permutation(labels)
        if permuted.sum() in (0, len(permuted)):  # guard degenerate shuffles
            continue
        if _cv_auc(features, permuted, n_splits, seed) >= observed:
            ge += 1
    p_value = ge / (n_permutations + 1)

    return OmnibusResult(
        auc=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        top_k=k,
        n_splits=n_splits,
    )


def group_mean_difference_shape(
    model: ShapeModel, is_cancer: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct each group's mean gland and their difference, as ``(M, 3)`` point sets.

    Each group mean is the overall mean shape plus that group's mean score projected back
    through the modes, so the result lives in the same physical-mm frame as ``model.mean``.

    Returns:
        ``(mean_healthy, mean_cancer, delta)`` where ``delta = mean_cancer - mean_healthy``;
        the magnitude of ``delta`` per point shows *where* on the gland the groups diverge.
    """
    groups = _as_bool_groups(is_cancer, model.scores.shape[0])

    def _reconstruct(mask: np.ndarray) -> np.ndarray:
        mean_score = model.scores[mask].mean(axis=0)  # (K,)
        offset = (mean_score @ model.components).reshape(model.n_points, 3)
        return model.mean + offset

    mean_healthy = _reconstruct(~groups)
    mean_cancer = _reconstruct(groups)
    return mean_healthy, mean_cancer, mean_cancer - mean_healthy


@dataclass
class VolumeDifference:
    """Descriptive gland-volume comparison between groups (cm^3)."""

    median_healthy: float
    median_cancer: float
    u_statistic: float
    p_value: float
    cohens_d: float
    n_healthy: int
    n_cancer: int


def volume_difference(cohort: list[CohortEntry]) -> VolumeDifference:
    """Mann-Whitney comparison of raw gland volume (cm^3) between groups.

    Reported separately from the shape test: when the model is built with ``gpa(scale=False)``
    a shape-mode difference may partly reflect size, so it helps to see the size gap on its own.
    """
    vols = np.array([e.volume_cm3 for e in cohort], dtype=np.float64)
    groups = _as_bool_groups(np.array([e.is_cancer for e in cohort]), len(cohort))
    cancer, healthy = vols[groups], vols[~groups]
    u, p = stats.mannwhitneyu(cancer, healthy, alternative="two-sided")
    return VolumeDifference(
        median_healthy=float(np.median(healthy)),
        median_cancer=float(np.median(cancer)),
        u_statistic=float(u),
        p_value=float(p),
        cohens_d=_cohens_d(cancer, healthy),
        n_healthy=int((~groups).sum()),
        n_cancer=int(groups.sum()),
    )


@dataclass
class GroupComparison:
    """Everything the healthy-vs-cancer analysis produces, in one object."""

    n_healthy: int
    n_cancer: int
    mode_tests: list[ModeTest]
    omnibus: OmnibusResult
    mean_healthy: np.ndarray  # (M, 3)
    mean_cancer: np.ndarray  # (M, 3)
    mean_delta: np.ndarray  # (M, 3) = mean_cancer - mean_healthy
    volume: VolumeDifference | None

    def significant_modes(self, alpha: float = 0.05) -> list[ModeTest]:
        """Modes whose FDR-adjusted p-value is below ``alpha``."""
        return [t for t in self.mode_tests if t.p_adjusted < alpha]


def compare_groups(
    model: ShapeModel,
    is_cancer: np.ndarray,
    *,
    cohort: list[CohortEntry] | None = None,
    top_k: int = 10,
    n_splits: int = 5,
    n_permutations: int = 1000,
    seed: int = 0,
) -> GroupComparison:
    """Run the full healthy-vs-cancer analysis and bundle the results.

    Args:
        model: the fitted :class:`ShapeModel` (its ``scores`` rows must align with ``is_cancer``
            and, if given, ``cohort``).
        is_cancer: boolean ``(S,)`` group labels.
        cohort: the cohort entries, used only for the descriptive volume comparison. Pass it to
            populate ``GroupComparison.volume``; omit to skip the size readout.
        top_k: leading modes used for the per-mode family and the omnibus classifier.
        n_splits / n_permutations / seed: passed to :func:`shape_predicts_cancer`.
    """
    groups = _as_bool_groups(is_cancer, model.scores.shape[0])
    mean_healthy, mean_cancer, delta = group_mean_difference_shape(model, groups)
    return GroupComparison(
        n_healthy=int((~groups).sum()),
        n_cancer=int(groups.sum()),
        mode_tests=per_mode_tests(model.scores, groups, top_k=top_k),
        omnibus=shape_predicts_cancer(
            model.scores,
            groups,
            top_k=top_k,
            n_splits=n_splits,
            n_permutations=n_permutations,
            seed=seed,
        ),
        mean_healthy=mean_healthy,
        mean_cancer=mean_cancer,
        mean_delta=delta,
        volume=volume_difference(cohort) if cohort is not None else None,
    )
