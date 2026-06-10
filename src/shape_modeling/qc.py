"""Quality control for MRI volumes and segmentation masks.

A bad input silently corrupts a Statistical Shape Model: a disconnected mask island, an
organ truncated at the field-of-view edge, wrong label values, or a tumor drawn outside
the gland all distort the mean shape and every principal mode. :func:`run_qc` inspects
every volume and mask in a :class:`~shape_modeling.io.reader.Dataset` and returns a
:class:`QCReport` of findings, each tagged with a :class:`Severity`.

QC is **non-destructive and advisory**: it never modifies data and never raises. Callers
decide what to exclude based on the report. Thresholds live in :class:`QCConfig` and
default to prostate-plausible values; pass a different config for other organs.

Checks fall in three groups:

* **Image integrity** -- finite values, positive spacing, anisotropy, non-constant MRI,
  geometry consistency across a case's volumes.
* **Mask validity** -- non-empty, sane label values, single connected component (organ),
  no FOV-border contact, plausible physical volume.
* **Cross-mask** -- tumor lies inside the organ; mask sits on tissue, not background air.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import SimpleITK as sitk

from shape_modeling.io.reader import Case, Dataset, LoadedVolume, VolumeType, geometry_matches

__all__ = [
    "QCConfig",
    "QCFinding",
    "QCReport",
    "Severity",
    "qc_case",
    "run_qc",
]


class Severity(str, Enum):
    """Severity of a QC finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


#: Ordering used to pick the "worst" severity in a group.
_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2}


@dataclass
class QCFinding:
    """A single QC observation about a case or one of its volumes."""

    case_id: str
    target: str  # filename of the volume, or "case" for case-level findings
    check: str  # short id, e.g. "mask.components"
    severity: Severity
    message: str
    value: float | int | str | None = None


@dataclass
class QCConfig:
    """Tunable thresholds. Defaults are prostate-plausible."""

    max_anisotropy_ratio: float = 8.0
    organ_volume_range_cm3: tuple[float, float] = (10.0, 200.0)
    organ_max_components: int = 1
    island_min_voxels: int = 10
    min_tumor_in_organ_fraction: float = 0.5
    min_mask_in_foreground_fraction: float = 0.5


@dataclass
class QCReport:
    """All findings from a QC pass, with convenience accessors and a printable table."""

    findings: list[QCFinding] = field(default_factory=list)
    case_ids: list[str] = field(default_factory=list)  # every case checked, in order

    def for_case(self, case_id: str) -> list[QCFinding]:
        return [f for f in self.findings if f.case_id == case_id]

    def by_severity(self, severity: Severity) -> list[QCFinding]:
        return [f for f in self.findings if f.severity == severity]

    def errors(self) -> list[QCFinding]:
        return self.by_severity(Severity.ERROR)

    def warnings(self) -> list[QCFinding]:
        return self.by_severity(Severity.WARNING)

    @property
    def passed(self) -> bool:
        """True if there are no ERROR-level findings."""
        return not self.errors()

    def summary(self) -> str:
        """A per-case counts table followed by every WARNING/ERROR in detail."""
        case_ids = list(self.case_ids)
        for f in self.findings:  # include any case present only in findings
            if f.case_id not in case_ids:
                case_ids.append(f.case_id)

        header = f"{'case':<12} {'info':<5} {'warn':<5} {'error':<6} {'worst':<8}"
        lines = [header, "-" * len(header)]
        for cid in case_ids:
            fs = self.for_case(cid)
            n_info = sum(f.severity == Severity.INFO for f in fs)
            n_warn = sum(f.severity == Severity.WARNING for f in fs)
            n_err = sum(f.severity == Severity.ERROR for f in fs)
            worst = (
                max((f.severity for f in fs), key=lambda s: _SEVERITY_RANK[s])
                if fs
                else Severity.INFO
            )
            verdict = worst.value if fs else "clean"
            lines.append(f"{cid:<12} {n_info:<5} {n_warn:<5} {n_err:<6} {verdict:<8}")
        lines.append("-" * len(header))
        verdict = "PASS (no errors)" if self.passed else f"FAIL ({len(self.errors())} error(s))"
        lines.append(f"{len(case_ids)} case(s); {len(self.warnings())} warning(s); {verdict}")

        notable = [f for f in self.findings if f.severity != Severity.INFO]
        if notable:
            lines.append("")
            lines.append("Findings (warnings + errors):")
            for f in notable:
                lines.append(
                    f"  [{f.severity.value:<7}] {f.case_id} · {f.target} · {f.check}: {f.message}"
                )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #


def _foreground(vol: LoadedVolume) -> np.ndarray:
    """Boolean foreground mask (nonzero voxels) for a loaded volume."""
    assert vol.array is not None, "QC requires arrays; load the dataset with load_arrays=True"
    return vol.array != 0


def _binary_components(vol: LoadedVolume) -> list[int]:
    """Voxel sizes of connected components of the foreground, largest first.

    Uses SimpleITK so component connectivity respects the 3-D voxel grid.
    """
    assert vol.sitk_image is not None
    binary = sitk.NotEqual(vol.sitk_image, 0)
    labeled = sitk.ConnectedComponent(binary)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labeled)
    sizes = [int(stats.GetNumberOfPixels(label)) for label in stats.GetLabels()]
    return sorted(sizes, reverse=True)


def _physical_volume_cm3(vol: LoadedVolume) -> float:
    """Physical volume of the foreground in cm³ (voxel count × voxel volume)."""
    voxel_mm3 = float(np.prod(vol.spacing))
    return int(_foreground(vol).sum()) * voxel_mm3 / 1000.0


def _touches_border(fg: np.ndarray) -> bool:
    """True if any foreground voxel lies on the first/last index of any axis."""
    for axis in range(fg.ndim):
        first = np.take(fg, 0, axis=axis)
        last = np.take(fg, fg.shape[axis] - 1, axis=axis)
        if first.any() or last.any():
            return True
    return False


# --------------------------------------------------------------------------- #
# Image-integrity checks (any volume)
# --------------------------------------------------------------------------- #


def _qc_volume(case_id: str, vol: LoadedVolume, config: QCConfig) -> list[QCFinding]:
    findings: list[QCFinding] = []
    name = vol.path.name
    is_mri = vol.volume_type == VolumeType.MRI

    if vol.array is not None and not np.isfinite(vol.array).all():
        findings.append(
            QCFinding(case_id, name, "image.finite", Severity.ERROR, "contains NaN/Inf voxels")
        )

    if any(s <= 0 for s in vol.spacing):
        findings.append(
            QCFinding(
                case_id,
                name,
                "image.spacing",
                Severity.ERROR,
                f"non-positive spacing {vol.spacing}",
            )
        )
    else:
        ratio = max(vol.spacing) / min(vol.spacing)
        if ratio > config.max_anisotropy_ratio:
            findings.append(
                QCFinding(
                    case_id,
                    name,
                    "image.anisotropy",
                    Severity.WARNING,
                    f"high voxel anisotropy (ratio {ratio:.1f})",
                    round(ratio, 2),
                )
            )

    if is_mri and vol.array is not None and vol.array.size:
        n_unique = int(len(np.unique(vol.array)))
        if n_unique == 1:
            value = float(vol.array.flat[0])
            sev = Severity.ERROR if value == 0 else Severity.WARNING
            findings.append(
                QCFinding(
                    case_id, name, "image.constant", sev, f"MRI is constant (value {value})", value
                )
            )

    return findings


# --------------------------------------------------------------------------- #
# Mask-validity checks (organ + cancer segmentations)
# --------------------------------------------------------------------------- #


def _qc_mask(case_id: str, vol: LoadedVolume, config: QCConfig) -> list[QCFinding]:
    findings: list[QCFinding] = []
    name = vol.path.name
    is_organ = vol.volume_type == VolumeType.ORGAN_SEG
    kind = "organ" if is_organ else "tumor"

    assert vol.array is not None
    fg = _foreground(vol)
    n_fg = int(fg.sum())

    # Empty mask.
    if n_fg == 0:
        sev = Severity.ERROR if is_organ else Severity.WARNING
        findings.append(
            QCFinding(case_id, name, "mask.empty", sev, f"{kind} mask has no foreground")
        )
        return findings  # nothing else to measure

    # Label values.
    labels = np.unique(vol.array)
    nonzero_labels = labels[labels != 0]
    non_integer = bool(np.any(nonzero_labels != np.round(nonzero_labels)))
    if non_integer:
        findings.append(
            QCFinding(
                case_id,
                name,
                "mask.labels",
                Severity.WARNING,
                f"non-integer label values {nonzero_labels[:5].tolist()} (probability map?)",
            )
        )

    # Connected components.
    sizes = _binary_components(vol)
    big = [s for s in sizes if s >= config.island_min_voxels]
    islands = [s for s in sizes if s < config.island_min_voxels]
    largest_frac = sizes[0] / n_fg if sizes else 0.0
    if is_organ and len(big) > config.organ_max_components:
        findings.append(
            QCFinding(
                case_id,
                name,
                "mask.components",
                Severity.WARNING,
                f"organ has {len(big)} components (largest holds {largest_frac:.0%})",
                len(big),
            )
        )
    elif not is_organ:
        findings.append(
            QCFinding(
                case_id,
                name,
                "mask.components",
                Severity.INFO,
                f"{len(big)} tumor focus/foci",
                len(big),
            )
        )
    if islands:
        findings.append(
            QCFinding(
                case_id,
                name,
                "mask.components",
                Severity.WARNING,
                f"{len(islands)} tiny island(s) < {config.island_min_voxels} voxels",
                len(islands),
            )
        )

    # FOV-border contact.
    if _touches_border(fg):
        findings.append(
            QCFinding(
                case_id,
                name,
                "mask.border",
                Severity.WARNING,
                f"{kind} mask touches the volume border (truncated FOV?)",
            )
        )

    # Physical volume.
    vol_cm3 = _physical_volume_cm3(vol)
    if is_organ:
        lo, hi = config.organ_volume_range_cm3
        if not (lo <= vol_cm3 <= hi):
            findings.append(
                QCFinding(
                    case_id,
                    name,
                    "mask.volume_cm3",
                    Severity.WARNING,
                    f"organ volume {vol_cm3:.1f} cm³ outside plausible [{lo:.0f}, {hi:.0f}]",
                    round(vol_cm3, 2),
                )
            )
    else:
        findings.append(
            QCFinding(
                case_id,
                name,
                "mask.volume_cm3",
                Severity.INFO,
                f"tumor volume {vol_cm3:.2f} cm³",
                round(vol_cm3, 3),
            )
        )

    return findings


# --------------------------------------------------------------------------- #
# Cross-mask checks (case level)
# --------------------------------------------------------------------------- #


def _matching_organ(tumor: LoadedVolume, organs: list[LoadedVolume]) -> LoadedVolume | None:
    """Pick an organ mask on the same grid as the tumor (prefer same sequence)."""
    compatible = [o for o in organs if geometry_matches(tumor, o)]
    if not compatible:
        return None
    for o in compatible:
        if o.sequence == tumor.sequence:
            return o
    return compatible[0]


def _qc_cross_mask(case: Case, config: QCConfig) -> list[QCFinding]:
    findings: list[QCFinding] = []
    cid = case.case_id

    # Tumor inside organ.
    for tumor in case.cancer_segs:
        if not case.organ_segs:
            findings.append(
                QCFinding(
                    cid,
                    tumor.path.name,
                    "cross.tumor_in_organ",
                    Severity.INFO,
                    "no organ mask to compare",
                )
            )
            continue
        organ = _matching_organ(tumor, case.organ_segs)
        if organ is None:
            findings.append(
                QCFinding(
                    cid,
                    tumor.path.name,
                    "cross.tumor_in_organ",
                    Severity.INFO,
                    "organ geometry differs; skipped",
                )
            )
            continue
        t_fg = _foreground(tumor)
        inside = int((t_fg & _foreground(organ)).sum())
        total = int(t_fg.sum())
        frac = inside / total if total else 0.0
        if frac < config.min_tumor_in_organ_fraction:
            findings.append(
                QCFinding(
                    cid,
                    tumor.path.name,
                    "cross.tumor_in_organ",
                    Severity.WARNING,
                    f"only {frac:.0%} of tumor lies inside the organ ({organ.path.name})",
                    round(frac, 3),
                )
            )

    # Mask sits on tissue, not background air.
    for mask in (*case.organ_segs, *case.cancer_segs):
        mri = case.mris.get(mask.sequence) if mask.sequence else None
        mri = mri or case.primary_mri
        if mri is None or mri.array is None or not geometry_matches(mask, mri):
            continue
        m_fg = _foreground(mask)
        if not m_fg.any():
            continue
        # Background = at or below a low percentile of the whole MRI.
        bg_level = float(np.percentile(mri.array, 5))
        on_tissue = int((m_fg & (mri.array > bg_level)).sum())
        frac = on_tissue / int(m_fg.sum())
        if frac < config.min_mask_in_foreground_fraction:
            findings.append(
                QCFinding(
                    cid,
                    mask.path.name,
                    "cross.mask_in_foreground",
                    Severity.WARNING,
                    f"only {frac:.0%} of the mask overlaps MRI tissue (rest is background)",
                    round(frac, 3),
                )
            )

    return findings


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def qc_case(case: Case, config: QCConfig | None = None) -> list[QCFinding]:
    """Run all QC checks for a single case and return its findings."""
    config = config or QCConfig()
    findings: list[QCFinding] = []

    if case.primary_mri is None:
        findings.append(
            QCFinding(case.case_id, "case", "image.geometry", Severity.ERROR, "case has no MRI")
        )

    all_volumes = [*case.mris.values(), *case.organ_segs, *case.cancer_segs]
    for vol in all_volumes:
        findings.extend(_qc_volume(case.case_id, vol, config))

    # Geometry consistency against the reference MRI.
    reference = case.primary_mri
    if reference is not None:
        for vol in all_volumes:
            if vol is not reference and not geometry_matches(reference, vol):
                findings.append(
                    QCFinding(
                        case.case_id,
                        vol.path.name,
                        "image.geometry",
                        Severity.ERROR,
                        f"geometry differs from reference MRI {reference.path.name}",
                    )
                )

    for mask in (*case.organ_segs, *case.cancer_segs):
        findings.extend(_qc_mask(case.case_id, mask, config))

    findings.extend(_qc_cross_mask(case, config))
    return findings


def run_qc(dataset: Dataset, config: QCConfig | None = None) -> QCReport:
    """Run QC across every case in ``dataset`` and return a :class:`QCReport`."""
    config = config or QCConfig()
    findings: list[QCFinding] = []
    for case in dataset.cases:
        findings.extend(qc_case(case, config))
    return QCReport(findings=findings, case_ids=[c.case_id for c in dataset.cases])
