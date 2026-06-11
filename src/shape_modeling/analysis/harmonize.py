"""ComBat batch-effect harmonization of corresponded particle coordinates.

The multi-center healthy basis is contaminated by *acquisition site* (receive coil, vendor,
field strength, resolution): the leading PCA modes separate by source, so the model encodes the
scanner as much as the anatomy. ComBat (Johnson et al. 2007, ported here from genomics via
``neuroCombat``) removes that with a per-batch location/scale model fit by empirical Bayes --
shrinking each batch's additive shift and multiplicative scale toward the pooled estimate, which
matters when the smallest sites have only ~12 subjects.

This is the *production* counterpart to the location/scale diagnostic in
``notebooks/explore_shapes.ipynb`` (section 6): harmonize the GPA-aligned **coordinates** and
re-fit GPA + PCA, rather than correcting the PCA scores after the fact. Real biological size
variation is preserved by passing raw centroid size as a **protected continuous covariate**, so
ComBat regresses it out of the batch model and adds it back -- site is removed without erasing
anatomy.

``neuroCombat`` is an optional dependency (``pip install neuroCombat``, part of the ``analysis``
extra); the import is deferred to call time so the rest of the package works without it, mirroring
the ShapeWorks handling in :mod:`~shape_modeling.analysis.correspondence`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["combat_harmonize"]


def combat_harmonize(
    particles: np.ndarray,
    batch: np.ndarray,
    continuous: np.ndarray | None = None,
    *,
    covariate_names: list[str] | None = None,
    ref_batch: str | None = None,
) -> np.ndarray:
    """ComBat-harmonize ``(S, M, 3)`` corresponded particle coordinates against ``batch``.

    Each of the ``3M`` coordinate features is corrected for the additive and multiplicative
    batch (site) effect with an empirical-Bayes location/scale model. The coordinates should be
    **GPA-aligned** (output of :func:`~shape_modeling.analysis.ssm.gpa`) so the batch effect is
    pose-free; re-run ``gpa`` + :func:`~shape_modeling.analysis.ssm.build_ssm` on the result to
    get the harmonized model.

    Args:
        particles: ``(S, M, 3)`` corresponded points (one row per subject, same point order).
        batch: ``(S,)`` site / acquisition labels (the effect to remove). Needs >= 2 levels.
        continuous: optional ``(S,)`` or ``(S, C)`` protected continuous covariate(s) -- e.g.
            raw (pre-GPA) centroid size. Regressed out of the batch model and added back, so
            genuine biological variation along it is **not** harmonized away.
        covariate_names: optional names for the continuous columns (for the ComBat design only).
        ref_batch: optional batch label to use as the reference (others are mapped onto it);
            ``None`` maps every batch onto the pooled grand mean/variance.

    Returns:
        ``(S, M, 3)`` harmonized coordinates, same shape and ordering as ``particles``.
    """
    arr = np.asarray(particles, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (S, M, 3), got {arr.shape}")
    n_subjects, n_points, _ = arr.shape

    batch = np.asarray(batch)
    if batch.shape != (n_subjects,):
        raise ValueError(f"batch must be ({n_subjects},), got {batch.shape}")
    if np.unique(batch).size < 2:
        raise ValueError("ComBat needs at least 2 batch levels to estimate a site effect")

    # neuroCombat expects features x samples, so transpose the flattened (S, 3M) coordinates.
    flat = arr.reshape(n_subjects, n_points * 3)

    try:
        from neuroCombat import neuroCombat
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "ComBat harmonization requires the optional `neuroCombat` package: "
            "pip install neuroCombat"
        ) from exc
    import pandas as pd

    covars: dict[str, np.ndarray] = {"batch": batch}
    continuous_cols: list[str] = []
    if continuous is not None:
        cont = np.asarray(continuous, dtype=np.float64)
        if cont.ndim == 1:
            cont = cont[:, None]
        if cont.shape[0] != n_subjects:
            raise ValueError(
                f"continuous must have {n_subjects} rows, got {cont.shape[0]}"
            )
        names = covariate_names or [f"cov{i}" for i in range(cont.shape[1])]
        if len(names) != cont.shape[1]:
            raise ValueError("covariate_names length must match continuous columns")
        for name, column in zip(names, cont.T, strict=True):
            covars[name] = column
            continuous_cols.append(name)

    result = neuroCombat(
        dat=flat.T,
        covars=pd.DataFrame(covars),
        batch_col="batch",
        continuous_cols=continuous_cols,
        ref_batch=ref_batch,
    )
    return np.ascontiguousarray(result["data"].T).reshape(n_subjects, n_points, 3)
