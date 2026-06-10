"""Exploratory shape-analysis stage: cohort selection, surfaces, correspondence, SSM.

This package turns a discovered :class:`~shape_modeling.io.reader.Dataset` into a
first-pass Statistical Shape Model so the cohort can be inspected before the final model
is built (pipeline stages 3--5). The pieces are deliberately lightweight and reusable:

* :mod:`~shape_modeling.analysis.cohort` -- select the cases to model and attach metadata.
* :mod:`~shape_modeling.analysis.surface` -- clean masks and extract surface meshes.
* :mod:`~shape_modeling.analysis.correspondence` -- ShapeWorks particle correspondence
  (optional; requires a separate ShapeWorks install).
* :mod:`~shape_modeling.analysis.ssm` -- generalized Procrustes alignment + PCA.

``correspondence`` is imported lazily so the rest of the package works without ShapeWorks.
"""

from __future__ import annotations

from shape_modeling.analysis.cohort import (
    CohortEntry,
    build_cohort,
    build_grouped_cohort,
    gland_volume_cm3,
    is_healthy,
    whole_gland_mask,
)
from shape_modeling.analysis.groups import (
    GroupComparison,
    ModeTest,
    OmnibusResult,
    VolumeDifference,
    compare_groups,
    group_mean_difference_shape,
    per_mode_tests,
    shape_predicts_cancer,
    volume_difference,
)
from shape_modeling.analysis.ssm import (
    ShapeModel,
    build_ssm,
    centroid_size,
    gpa,
    mode_shape,
)
from shape_modeling.analysis.surface import (
    CleanConfig,
    Surface,
    clean_mask,
    export_mode_shapes,
    extract_surface,
    hull_surface,
    save_ply,
    warp_surface,
)

__all__ = [
    "CleanConfig",
    "CohortEntry",
    "GroupComparison",
    "ModeTest",
    "OmnibusResult",
    "ShapeModel",
    "Surface",
    "VolumeDifference",
    "build_cohort",
    "build_grouped_cohort",
    "build_ssm",
    "centroid_size",
    "clean_mask",
    "compare_groups",
    "export_mode_shapes",
    "extract_surface",
    "gland_volume_cm3",
    "gpa",
    "group_mean_difference_shape",
    "hull_surface",
    "is_healthy",
    "mode_shape",
    "per_mode_tests",
    "save_ply",
    "shape_predicts_cancer",
    "volume_difference",
    "warp_surface",
    "whole_gland_mask",
]


def __getattr__(name: str) -> object:  # pragma: no cover - thin lazy re-export
    """Lazily expose the ShapeWorks-dependent API without importing it at package load."""
    if name in {"ShapeWorksConfig", "ParticleSystem", "run_shapeworks"}:
        from shape_modeling.analysis import correspondence

        return getattr(correspondence, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
