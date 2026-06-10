"""Select the cases to model and attach the metadata used to colour the shape space.

The exploratory SSM is built from **healthy whole-gland** shapes (:func:`build_cohort`): every
case with no cancer segmentation, drawn from one or more discovered datasets. Each kept case
becomes a :class:`CohortEntry` carrying its source dataset and gland volume so the PCA scatter
can be coloured by provenance and size.

For the healthy-vs-cancer group analysis (stage 6, :func:`build_grouped_cohort`) we keep
**both** groups instead and tag each entry with ``is_cancer``. In both cases the modelled
geometry is the **whole gland** -- a cancer case's *lesion* mask only sets its group label,
never its shape.

Conventions (see ``CLAUDE.md``): whole gland means ``organ mask > 0`` (comparable across
datasets that store sub-region labels), and when several reader annotations exist we use the
first (``organ_segs[0]``). Cases with no organ segmentation cannot be modelled and are
skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shape_modeling.io.reader import Case, Dataset, LoadedVolume

__all__ = [
    "CohortEntry",
    "build_cohort",
    "build_grouped_cohort",
    "gland_volume_cm3",
    "is_healthy",
    "whole_gland_mask",
]


@dataclass
class CohortEntry:
    """One subject in the shape cohort, with the metadata used for inspection."""

    case_id: str
    source: str  # dataset label, e.g. "PROMISE12" or "Prostate158"
    volume_cm3: float
    organ: LoadedVolume  # the chosen whole-gland segmentation (geometry + pixels)
    is_cancer: bool = False  # set by build_grouped_cohort; False for the healthy cohort


def is_healthy(case: Case) -> bool:
    """True if the case has no cancer segmentation.

    The reader already drops all-zero ``empty`` placeholder tumour files via
    ``IGNORE_TOKENS``, so a "no tumour" case correctly arrives here with empty
    ``cancer_segs``.
    """
    return not case.cancer_segs


def whole_gland_mask(case: Case) -> tuple[np.ndarray, LoadedVolume]:
    """Return the binary whole-gland mask (``organ > 0``) and its source volume.

    Uses the first organ segmentation (``organ_segs[0]``). Raises if the case has no organ
    segmentation or its pixels were not loaded (``discover_cases(..., load_arrays=True)``).
    """
    if not case.organ_segs:
        raise ValueError(f"case {case.case_id!r} has no organ segmentation")
    organ = case.organ_segs[0]
    if organ.array is None:
        raise ValueError(
            f"case {case.case_id!r}: organ pixels not loaded; "
            f"call discover_cases(..., load_arrays=True)"
        )
    return organ.array > 0, organ


def gland_volume_cm3(volume: LoadedVolume) -> float:
    """Physical volume of the foreground (``> 0``) of a mask, in cm^3."""
    if volume.array is None:
        raise ValueError(f"{volume.path.name}: pixels not loaded")
    voxel_mm3 = math.prod(volume.spacing)
    return float((volume.array > 0).sum()) * voxel_mm3 / 1000.0


def build_cohort(datasets: dict[str, Dataset]) -> list[CohortEntry]:
    """Build the healthy whole-gland cohort from one or more labelled datasets.

    Args:
        datasets: maps a **source label** (used to colour the shape space, e.g.
            ``"PROMISE12"``) to a discovered :class:`Dataset`. Discover each dataset
            separately so the label is unambiguous.

    Returns:
        One :class:`CohortEntry` per healthy case that has an organ segmentation, sorted by
        ``(source, case_id)``. Cancer cases and cases without an organ mask are skipped.
    """
    entries: list[CohortEntry] = []
    for source, dataset in datasets.items():
        for case in dataset.cases:
            if not is_healthy(case) or not case.organ_segs:
                continue
            organ = case.organ_segs[0]
            if organ.array is None:
                continue
            entries.append(
                CohortEntry(
                    case_id=case.case_id,
                    source=source,
                    volume_cm3=gland_volume_cm3(organ),
                    organ=organ,
                )
            )
    entries.sort(key=lambda e: (e.source, e.case_id))
    return entries


def build_grouped_cohort(datasets: dict[str, Dataset]) -> list[CohortEntry]:
    """Build the **healthy-and-cancer** whole-gland cohort for the group analysis (stage 6).

    Like :func:`build_cohort` but keeps *every* case that has an organ segmentation, tagging
    each entry with ``is_cancer = bool(case.cancer_segs)``. The geometry is still the whole
    gland (``organ_segs[0] > 0``); the lesion mask only decides the group label, so the two
    groups are directly comparable in shape space. For an unconfounded comparison, pass a
    **single** dataset whose healthy and cancer cases share a segmentation protocol (e.g.
    ``{"Prostate158": ds}``) rather than pooling datasets with different gland-mask sources.

    Args:
        datasets: maps a **source label** to a discovered :class:`Dataset`.

    Returns:
        One :class:`CohortEntry` per case with an organ segmentation, sorted by
        ``(source, is_cancer, case_id)``. Cases without an organ mask are skipped.
    """
    entries: list[CohortEntry] = []
    for source, dataset in datasets.items():
        for case in dataset.cases:
            if not case.organ_segs:
                continue
            organ = case.organ_segs[0]
            if organ.array is None:
                continue
            entries.append(
                CohortEntry(
                    case_id=case.case_id,
                    source=source,
                    volume_cm3=gland_volume_cm3(organ),
                    organ=organ,
                    is_cancer=not is_healthy(case),
                )
            )
    entries.sort(key=lambda e: (e.source, e.is_cancer, e.case_id))
    return entries
