"""Tests for shape_modeling.qc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from shape_modeling.io.reader import Case, LoadedVolume, VolumeType, discover_cases
from shape_modeling.qc import QCConfig, Severity, qc_case, run_qc

SPACING = (1.0, 1.0, 1.0)  # 1 mm³ voxels -> volume in cm³ == voxel count / 1000

# The synthetic organ blob is ~2.5 cm³, so use a fixture-scale plausible range. (The
# QCConfig defaults are tuned for a real prostate in mm.)
CFG = QCConfig(organ_volume_range_cm3=(1.0, 50.0))


def _write(path: Path, arr: np.ndarray, spacing: tuple[float, float, float] = SPACING) -> None:
    img = sitk.GetImageFromArray(arr)  # arr is [z, y, x]
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))


def _mri(shape=(20, 40, 40), value=500) -> np.ndarray:
    """A non-constant, finite MRI: a smooth ramp with a bright box on top."""
    ramp = np.broadcast_to(np.linspace(50, 150, shape[2], dtype=np.float32), shape)
    arr = ramp.copy()
    arr[:, 8:32, 8:32] += value
    return arr


def _organ(shape=(20, 40, 40)) -> np.ndarray:
    """A single solid organ blob (well inside the FOV)."""
    arr = np.zeros(shape, dtype=np.uint8)
    arr[5:15, 12:28, 12:28] = 1
    return arr


def _clean_case(tmp_path: Path) -> Path:
    """A patient dir with a clean MRI + organ + tumor (tumor inside organ)."""
    d = tmp_path / "021"
    d.mkdir()
    _write(d / "t2.nii.gz", _mri())
    _write(d / "t2_anatomy_reader1.nii.gz", _organ())
    tumor = np.zeros((20, 40, 40), dtype=np.float32)
    tumor[8:12, 16:22, 16:22] = 3  # fully within the organ box
    _write(d / "t2_tumor_reader1.nii.gz", tumor)
    return d


def _checks(report, case_id: str) -> set[tuple[str, Severity]]:
    return {(f.check, f.severity) for f in report.for_case(case_id)}


def test_clean_case_passes(tmp_path: Path) -> None:
    report = run_qc(discover_cases(_clean_case(tmp_path)), CFG)
    assert report.passed
    assert not report.warnings()


def test_nan_mri_flagged() -> None:
    # SimpleITK's NIfTI writer drops NaN, so build the volume in memory to exercise the
    # finite check (it still guards real arrays loaded from other sources/formats).
    arr = _mri()
    arr[0, 0, 0] = np.nan
    mri = LoadedVolume(
        path=Path("t2.nii.gz"),
        volume_type=VolumeType.MRI,
        size=(40, 40, 20),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        dtype="32-bit float",
        sequence="T2",
        mri_type="T2",
        array=arr,
    )
    findings = qc_case(Case(case_id="021", mris={"T2": mri}), CFG)
    checks = {(f.check, f.severity) for f in findings}
    assert ("image.finite", Severity.ERROR) in checks


def test_empty_organ_is_error(tmp_path: Path) -> None:
    d = _clean_case(tmp_path)
    _write(d / "t2_anatomy_reader1.nii.gz", np.zeros((20, 40, 40), dtype=np.uint8))
    report = run_qc(discover_cases(d), CFG)
    assert ("mask.empty", Severity.ERROR) in _checks(report, "021")


def test_organ_island_flagged(tmp_path: Path) -> None:
    d = _clean_case(tmp_path)
    organ = _organ()
    organ[18, 36, 36] = 1  # a lone detached voxel far from the blob
    # one stray voxel < island_min_voxels -> island warning, and 2 components
    _write(d / "t2_anatomy_reader1.nii.gz", organ)
    report = run_qc(discover_cases(d), CFG)
    checks = _checks(report, "021")
    assert ("mask.components", Severity.WARNING) in checks


def test_organ_border_contact_flagged(tmp_path: Path) -> None:
    d = _clean_case(tmp_path)
    organ = np.zeros((20, 40, 40), dtype=np.uint8)
    organ[5:15, 0:16, 12:28] = 1  # touches y=0 border
    _write(d / "t2_anatomy_reader1.nii.gz", organ)
    report = run_qc(discover_cases(d), CFG)
    assert ("mask.border", Severity.WARNING) in _checks(report, "021")


def test_organ_volume_out_of_range_flagged(tmp_path: Path) -> None:
    d = _clean_case(tmp_path)
    # Tight plausible range so the ~2.5 cm³ test organ is flagged as too small.
    report = run_qc(discover_cases(d), QCConfig(organ_volume_range_cm3=(50.0, 200.0)))
    assert ("mask.volume_cm3", Severity.WARNING) in _checks(report, "021")


def test_tumor_outside_organ_flagged(tmp_path: Path) -> None:
    d = _clean_case(tmp_path)
    tumor = np.zeros((20, 40, 40), dtype=np.float32)
    tumor[8:12, 0:4, 0:4] = 3  # corner, outside the organ box
    _write(d / "t2_tumor_reader1.nii.gz", tumor)
    report = run_qc(discover_cases(d), CFG)
    assert ("cross.tumor_in_organ", Severity.WARNING) in _checks(report, "021")


def test_organ_volume_too_small_with_defaults(tmp_path: Path) -> None:
    # With the prostate-scale defaults, the ~2.5 cm³ synthetic organ is implausible.
    report = run_qc(discover_cases(_clean_case(tmp_path)))  # default QCConfig
    assert ("mask.volume_cm3", Severity.WARNING) in _checks(report, "021")


def test_summary_renders(tmp_path: Path) -> None:
    report = run_qc(discover_cases(_clean_case(tmp_path)), CFG)
    out = report.summary()
    assert "021" in out
    assert "PASS" in out
