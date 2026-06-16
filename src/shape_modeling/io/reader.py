"""Discover, classify, and load MRI volumes and segmentation masks.

The reader scans a directory of medical image files (MetaImage ``.mhd``, NIfTI
``.nii``/``.nii.gz``, or NRRD ``.nrrd``/``.nhdr``), groups them into per-subject
:class:`Case` objects, and works out
for each file whether it is an MRI volume, an organ ("anatomy") segmentation, or a
cancer ("tumor") segmentation.

Two data layouts are supported transparently:

* **Per-patient directory** -- one folder per case (e.g. ``021/``) holding several MRI
  sequences (``t2.nii.gz``, ``adc.nii.gz``, ``dwi.nii.gz``) and segmentations named
  ``<sequence>_<structure>_reader<N>.nii.gz`` (e.g. ``t2_anatomy_reader1.nii.gz``,
  ``adc_tumor_reader2.nii.gz``). The case id is the directory name; a case can hold
  multiple MRI sequences and multiple reader annotations per structure.
* **Flat** -- many cases in one folder, the case id encoded in each filename
  (e.g. PROMISE12 ``Case38.mhd`` + ``Case38_segmentation.mhd``).

Classification is **filename-token driven**: a name part of ``tumor``/``lesion`` marks a
cancer segmentation, ``anatomy``/``segmentation`` an organ segmentation, ``reader`` a
manual segmentation, and a bare sequence name (``t2``/``adc``/``dwi``...) an MRI. This is
authoritative over pixel dtype, because tumor masks are sometimes stored as ``float`` yet
are still binary label maps. Pixel content is only used for soft sanity warnings.

Two deliberate limitations: the MRI **sequence type** is taken from the filename (or
``default_mri_type`` when absent), since it is not stored in ``.mhd``/NIfTI headers; and
geometry mismatches between an MRI and its masks are **flagged, not corrected** --
resampling belongs to a later pipeline stage.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import SimpleITK as sitk

__all__ = [
    "Case",
    "Dataset",
    "LoadedVolume",
    "VolumeType",
    "classify_file",
    "detect_reader",
    "detect_sequence",
    "discover_cases",
    "geometry_matches",
    "load_volume",
    "parse_case_id",
]

#: File extensions SimpleITK can read as a single volume. ``.raw``/``.zraw`` are the
#: payloads of an ``.mhd`` header and are pulled in implicitly, so they are not globbed.
#: ``.nrrd``/``.nhdr`` (NRRD, detached or attached header) are common segmentation-export
#: formats (3D Slicer, NCI-ISBI) -- SimpleITK reads them like any other volume.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".mhd", ".nii", ".nii.gz", ".nrrd", ".nhdr")


class VolumeType(str, Enum):
    """Role of a volume within a case."""

    MRI = "mri"
    ORGAN_SEG = "organ_seg"
    CANCER_SEG = "cancer_seg"
    UNKNOWN = "unknown"


#: Filename name-parts that mark a cancer/lesion segmentation.
CANCER_TOKENS: frozenset[str] = frozenset({"tumor", "tumour", "lesion", "cancer"})

#: Filename name-parts that mark an organ/anatomy segmentation.
ORGAN_TOKENS: frozenset[str] = frozenset(
    {"anatomy", "segmentation", "seg", "mask", "label", "labels", "gt", "organ"}
)

#: Files whose name contains one of these parts are skipped entirely. ``empty`` marks an
#: all-zero placeholder volume (e.g. "no tumor for this patient"), which is not meaningful
#: for modeling.
IGNORE_TOKENS: frozenset[str] = frozenset({"empty"})

#: Filename name-parts that hint at an MRI sequence, mapped to a canonical label.
SEQUENCE_TOKENS: dict[str, str] = {
    "t2w": "T2W",
    "t2": "T2",
    "t1w": "T1W",
    "t1": "T1",
    "adc": "ADC",
    "dwi": "DWI",
    "dce": "DCE",
    "flair": "FLAIR",
}

#: Matches an annotator part such as ``reader1`` / ``reader`` / ``readers2``.
_READER_RE = re.compile(r"^readers?\d*$", re.IGNORECASE)

#: Splits a filename stem into its underscore/dash/space-separated parts.
_PART_RE = re.compile(r"[_\-\s]+")

#: A label mask is expected to have at most this many distinct values (any dtype). Used
#: only for soft warnings -- the filename token decides the role.
_MAX_MASK_LABELS = 256


@dataclass
class LoadedVolume:
    """A single image file with its role, sequence, annotator, and geometry.

    ``array`` and ``sitk_image`` are populated only when loaded with ``load_array=True``;
    otherwise just the header-derived metadata is available.
    """

    path: Path
    volume_type: VolumeType
    size: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]
    dtype: str
    sequence: str | None = None
    reader: str | None = None
    mri_type: str | None = None
    num_labels: int | None = None
    intensity_range: tuple[float, float] | None = None
    sitk_image: sitk.Image | None = None
    array: np.ndarray | None = None


@dataclass
class Case:
    """All volumes belonging to one subject.

    A case may hold several MRI sequences (keyed by sequence label in ``mris``) and
    several segmentations per structure (e.g. one per reader) in ``organ_segs`` and
    ``cancer_segs``.
    """

    case_id: str
    mris: dict[str, LoadedVolume] = field(default_factory=dict)
    organ_segs: list[LoadedVolume] = field(default_factory=list)
    cancer_segs: list[LoadedVolume] = field(default_factory=list)
    extras: list[LoadedVolume] = field(default_factory=list)
    geometry_consistent: bool = True

    @property
    def primary_mri(self) -> LoadedVolume | None:
        """The reference MRI: prefer T2/T2W, else the first one discovered."""
        if not self.mris:
            return None
        for key in ("T2W", "T2"):
            if key in self.mris:
                return self.mris[key]
        return next(iter(self.mris.values()))


@dataclass
class Dataset:
    """A collection of cases discovered under a directory."""

    cases: list[Case] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def summary(self) -> str:
        """Return a human-readable table of what was found per case."""
        header = f"{'case':<12} {'sequences':<18} {'organ':<6} {'cancer':<7} {'geom':<5}"
        lines = [header, "-" * len(header)]
        for case in self.cases:
            seqs = ",".join(case.mris) if case.mris else "-"
            lines.append(
                f"{case.case_id:<12} "
                f"{seqs:<18} "
                f"{len(case.organ_segs):<6} "
                f"{len(case.cancer_segs):<7} "
                f"{'ok' if case.geometry_consistent else 'MISMATCH':<5}"
            )
        n_cancer = sum(1 for c in self.cases if c.cancer_segs)
        lines.append("-" * len(header))
        lines.append(f"{len(self.cases)} case(s); {n_cancer} with cancer segmentation")
        return "\n".join(lines)


def _split_extension(filename: str) -> tuple[str, str]:
    """Split a filename into (stem, extension), handling the double ``.nii.gz``."""
    lower = filename.lower()
    for ext in SUPPORTED_EXTENSIONS:
        if lower.endswith(ext):
            return filename[: -len(ext)], filename[-len(ext) :]
    return Path(filename).stem, Path(filename).suffix


def _stem_parts(filename: str) -> list[str]:
    """Return the lower-cased name-parts of a filename's stem."""
    stem, _ = _split_extension(Path(filename).name)
    return [p for p in _PART_RE.split(stem) if p]


def _is_ignored(filename: str) -> bool:
    """True if the filename contains an ignore token (e.g. an ``empty`` placeholder)."""
    return any(p.lower() in IGNORE_TOKENS for p in _stem_parts(filename))


def _is_role_or_sequence_token(part: str) -> bool:
    low = part.lower()
    return (
        low in CANCER_TOKENS
        or low in ORGAN_TOKENS
        or low in SEQUENCE_TOKENS
        or bool(_READER_RE.match(low))
    )


def parse_case_id(filename: str) -> str:
    """Return the case id by dropping role, sequence, and reader name-parts.

    ``Case38_segmentation.mhd`` -> ``Case38``. A filename made up entirely of
    sequence/structure/reader tokens (e.g. ``t2_anatomy_reader1.nii.gz``) yields ``""``;
    the caller then falls back to the containing directory name.
    """
    stem, _ = _split_extension(Path(filename).name)
    parts = [p for p in _PART_RE.split(stem) if p]
    leftover = [p for p in parts if not _is_role_or_sequence_token(p)]
    return "_".join(leftover)


def detect_sequence(filename: str) -> str | None:
    """Return the canonical MRI sequence label from filename parts, or ``None``."""
    for part in _stem_parts(filename):
        canonical = SEQUENCE_TOKENS.get(part.lower())
        if canonical is not None:
            return canonical
    return None


def detect_reader(filename: str) -> str | None:
    """Return the annotator part (e.g. ``reader1``) from a filename, or ``None``."""
    for part in _stem_parts(filename):
        if _READER_RE.match(part):
            return part.lower()
    return None


def classify_file(filename: str) -> VolumeType:
    """Classify a file's role from its name parts.

    A ``tumor``/``lesion`` part means cancer; ``anatomy``/``segmentation`` (or any other
    organ token) means organ; a ``reader`` part with no structure token defaults to an
    organ segmentation; anything else is treated as an MRI. Content is not consulted here
    -- :func:`load_volume` may add a soft warning if the pixels disagree.
    """
    parts = {p.lower() for p in _stem_parts(filename)}
    if parts & CANCER_TOKENS:
        return VolumeType.CANCER_SEG
    if parts & ORGAN_TOKENS:
        return VolumeType.ORGAN_SEG
    if any(_READER_RE.match(p) for p in parts):
        return VolumeType.ORGAN_SEG
    return VolumeType.MRI


def load_volume(
    path: str | Path,
    *,
    load_array: bool = True,
    default_mri_type: str | None = None,
) -> LoadedVolume:
    """Load one volume's metadata (and optionally pixel data) and classify its role.

    The header is read first (cheap, no pixel data). When ``load_array`` is true the full
    image is loaded and label counts / intensity ranges are recorded; a soft warning is
    emitted if the content looks inconsistent with the filename-derived role, but the
    role is **not** changed (the filename is authoritative).
    """
    path = Path(path)
    name = path.name

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    pixel_id = reader.GetPixelID()

    volume_type = classify_file(name)
    sequence = detect_sequence(name)
    is_mri = volume_type == VolumeType.MRI

    vol = LoadedVolume(
        path=path,
        volume_type=volume_type,
        size=tuple(reader.GetSize()),
        spacing=tuple(reader.GetSpacing()),
        origin=tuple(reader.GetOrigin()),
        direction=tuple(reader.GetDirection()),
        dtype=sitk.GetPixelIDValueAsString(pixel_id),
        sequence=sequence or (default_mri_type if is_mri else None),
        reader=detect_reader(name),
    )
    if is_mri:
        vol.mri_type = vol.sequence

    if not load_array:
        return vol

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    vol.sitk_image = image
    vol.array = array

    if array.size == 0:
        return vol

    n_unique = int(len(np.unique(array)))
    if is_mri:
        vol.intensity_range = (float(array.min()), float(array.max()))
        if n_unique <= 4:
            warnings.warn(
                f"{name} is classified as MRI but has only {n_unique} distinct values; "
                f"it may actually be a label mask.",
                stacklevel=2,
            )
    else:
        vol.num_labels = n_unique
        if n_unique > _MAX_MASK_LABELS:
            warnings.warn(
                f"{name} is named like a segmentation but has {n_unique} distinct values; "
                f"it may be a probability map rather than a label mask.",
                stacklevel=2,
            )

    return vol


def geometry_matches(a: LoadedVolume, b: LoadedVolume, *, tol: float = 1e-3) -> bool:
    """True if two volumes share size, spacing, and orientation within tolerance."""
    if a.size != b.size:
        return False
    if not np.allclose(a.spacing, b.spacing, atol=tol):
        return False
    if len(a.direction) != len(b.direction) or not np.allclose(a.direction, b.direction, atol=tol):
        return False
    return True


def discover_cases(
    data_dir: str | Path,
    *,
    load_arrays: bool = True,
    default_mri_type: str | None = None,
    recursive: bool = True,
) -> Dataset:
    """Scan ``data_dir``, group files into cases, classify roles, and load volumes.

    Args:
        data_dir: A patient directory (its name becomes the case id), a parent of patient
            directories (one case each), or a flat folder whose filenames carry case ids.
        load_arrays: Load pixel data (enables label counts and intensity ranges). Set
            false for a fast metadata-only scan.
        default_mri_type: Sequence label for MRIs whose filename gives no hint
            (e.g. ``"T2W"`` for PROMISE12).
        recursive: Recurse into subdirectories (needed for the per-patient-dir layout).

    Returns:
        A :class:`Dataset` of :class:`Case` objects sorted by case id.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"{data_dir} is not a directory")

    paths: list[Path] = []
    iterator = data_dir.rglob("*") if recursive else data_dir.iterdir()
    for p in iterator:
        if not (p.is_file() and p.name.lower().endswith(SUPPORTED_EXTENSIONS)):
            continue
        if _is_ignored(p.name):
            continue
        paths.append(p)

    # Decide the layout per directory. If any file in a directory has no case id of its
    # own (a pure sequence/structure name like ``t2.nii.gz``), the whole directory is one
    # patient case and *every* file in it -- including oddly-named extras -- belongs to
    # that case. Otherwise the directory is a flat collection whose filenames carry the
    # case ids (e.g. PROMISE12 ``Case38.mhd``).
    by_dir: dict[Path, list[Path]] = {}
    for p in sorted(paths):
        by_dir.setdefault(p.parent, []).append(p)

    grouped: dict[str, list[Path]] = {}
    for dir_path, files in by_dir.items():
        ids = {f: parse_case_id(f.name) for f in files}
        is_per_patient_dir = any(cid == "" for cid in ids.values())
        # Flat-layout case ids live in the filename, so the same ``CaseNN`` can recur in
        # different sub-folders for *different* patients (PROMISE12 ``TestData`` vs
        # ``TrainingData``). Namespace them by the sub-folder relative to the scan root so
        # they stay distinct; a plain flat folder (files directly under ``data_dir``) keeps
        # its bare ids.
        rel = dir_path.relative_to(data_dir)
        for f in files:
            if is_per_patient_dir:
                case_id = dir_path.name
            elif rel == Path("."):
                case_id = ids[f]
            else:
                case_id = f"{rel.as_posix()}/{ids[f]}"
            grouped.setdefault(case_id, []).append(f)

    cases: list[Case] = []
    for case_id in sorted(grouped):
        case = Case(case_id=case_id)
        for p in grouped[case_id]:
            vol = load_volume(p, load_array=load_arrays, default_mri_type=default_mri_type)
            _assign_volume(case, vol)
        _check_case_geometry(case)
        cases.append(case)

    return Dataset(cases=cases)


def _assign_volume(case: Case, vol: LoadedVolume) -> None:
    """Place a loaded volume into the right collection on the case."""
    if vol.volume_type == VolumeType.MRI:
        key = vol.sequence or vol.path.stem
        if key in case.mris:
            warnings.warn(
                f"case {case.case_id} has more than one MRI for sequence '{key}' "
                f"({case.mris[key].path.name}, {vol.path.name}); keeping the first.",
                stacklevel=2,
            )
            case.extras.append(vol)
        else:
            case.mris[key] = vol
    elif vol.volume_type == VolumeType.ORGAN_SEG:
        case.organ_segs.append(vol)
    elif vol.volume_type == VolumeType.CANCER_SEG:
        case.cancer_segs.append(vol)
    else:
        case.extras.append(vol)


def _check_case_geometry(case: Case) -> None:
    """Flag the case if any volume's geometry differs from the reference MRI's."""
    reference = case.primary_mri
    if reference is None:
        return
    others = [
        v
        for v in (*case.mris.values(), *case.organ_segs, *case.cancer_segs)
        if v is not reference
    ]
    for vol in others:
        if not geometry_matches(reference, vol):
            case.geometry_consistent = False
            warnings.warn(
                f"case {case.case_id}: geometry of {vol.path.name} does not match the "
                f"reference MRI {reference.path.name} (size/spacing/orientation differ).",
                stacklevel=2,
            )
