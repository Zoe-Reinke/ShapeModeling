"""Tests for shape_modeling.io.reader."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from shape_modeling.io.reader import (
    VolumeType,
    classify_file,
    detect_reader,
    detect_sequence,
    discover_cases,
    load_volume,
    parse_case_id,
)

# --------------------------------------------------------------------------- #
# Pure-logic unit tests (no I/O)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # Flat layout (PROMISE12): the case id lives in the filename.
        ("Case38.mhd", "Case38"),
        ("Case38_segmentation.mhd", "Case38"),
        ("Case38_seg.nii.gz", "Case38"),
        # Per-patient-dir layout: pure sequence/structure/reader tokens -> empty,
        # so discover_cases falls back to the directory name.
        ("t2.nii.gz", ""),
        ("adc.nii.gz", ""),
        ("t2_anatomy_reader1.nii.gz", ""),
        ("adc_tumor_reader2.nii.gz", ""),
    ],
)
def test_parse_case_id(filename: str, expected: str) -> None:
    assert parse_case_id(filename) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # Sequence-only names are MRIs.
        ("t2.nii.gz", VolumeType.MRI),
        ("adc.nii.gz", VolumeType.MRI),
        ("dwi.nii.gz", VolumeType.MRI),
        ("Case38.mhd", VolumeType.MRI),
        # Anatomy / generic segmentation tokens -> organ.
        ("t2_anatomy_reader1.nii.gz", VolumeType.ORGAN_SEG),
        ("Case38_segmentation.mhd", VolumeType.ORGAN_SEG),
        ("Case38_mask.nii", VolumeType.ORGAN_SEG),
        # Tumor / lesion tokens -> cancer (even alongside a sequence + reader).
        ("t2_tumor_reader1.nii.gz", VolumeType.CANCER_SEG),
        ("adc_tumor_reader2.nii.gz", VolumeType.CANCER_SEG),
        ("Case38_lesion.nii.gz", VolumeType.CANCER_SEG),
    ],
)
def test_classify_file(filename: str, expected: VolumeType) -> None:
    assert classify_file(filename) == expected


def test_tumor_token_wins_over_sequence_and_reader() -> None:
    assert classify_file("adc_tumor_reader2.nii.gz") == VolumeType.CANCER_SEG


@pytest.mark.parametrize(
    ("filename", "sequence", "reader"),
    [
        ("t2_anatomy_reader1.nii.gz", "T2", "reader1"),
        ("adc_tumor_reader2.nii.gz", "ADC", "reader2"),
        ("dwi.nii.gz", "DWI", None),
        ("Case38.mhd", None, None),
    ],
)
def test_detect_sequence_and_reader(
    filename: str, sequence: str | None, reader: str | None
) -> None:
    assert detect_sequence(filename) == sequence
    assert detect_reader(filename) == reader


# --------------------------------------------------------------------------- #
# Hermetic integration tests with synthetic fixtures
# --------------------------------------------------------------------------- #


def _write_mri(path: Path, size: tuple[int, int, int] = (8, 8, 4)) -> None:
    """Write a continuous-intensity volume."""
    arr = np.arange(np.prod(size), dtype=np.int16).reshape(size[::-1])  # z,y,x
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.625, 0.625, 3.6))
    sitk.WriteImage(img, str(path))


def _write_mask(
    path: Path,
    size: tuple[int, int, int] = (8, 8, 4),
    *,
    dtype: type = np.uint8,
    label: float = 1,
) -> None:
    """Write a small-label mask (any dtype -- tumor masks are sometimes float)."""
    arr = np.zeros(size[::-1], dtype=dtype)
    arr[..., : size[0] // 2] = label
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.625, 0.625, 3.6))
    sitk.WriteImage(img, str(path))


def test_flat_layout_promise12_style(tmp_path: Path) -> None:
    _write_mri(tmp_path / "Case01.mhd")
    _write_mask(tmp_path / "Case01_segmentation.mhd")
    _write_mri(tmp_path / "Case02.mhd")
    _write_mask(tmp_path / "Case02_segmentation.mhd")

    ds = discover_cases(tmp_path, default_mri_type="T2W")
    assert len(ds) == 2
    case = ds.cases[0]
    assert set(case.mris) == {"T2W"}
    assert case.primary_mri is not None and case.primary_mri.mri_type == "T2W"
    assert len(case.organ_segs) == 1
    assert not case.cancer_segs
    assert case.geometry_consistent


def test_per_patient_dir_layout(tmp_path: Path) -> None:
    """A directory like 021/ -> one case with multiple sequences and readers."""
    patient = tmp_path / "021"
    patient.mkdir()
    _write_mri(patient / "t2.nii.gz")
    _write_mri(patient / "adc.nii.gz")
    _write_mri(patient / "dwi.nii.gz")
    _write_mask(patient / "t2_anatomy_reader1.nii.gz", label=2)  # organ
    _write_mask(patient / "t2_tumor_reader1.nii.gz", dtype=np.float32, label=3)  # cancer
    _write_mask(patient / "adc_tumor_reader1.nii.gz", dtype=np.float32, label=1)  # cancer
    _write_mask(patient / "adc_tumor_reader2.nii.gz", dtype=np.float64, label=1)  # cancer

    ds = discover_cases(patient)
    assert len(ds) == 1
    case = ds.cases[0]
    assert case.case_id == "021"
    assert set(case.mris) == {"T2", "ADC", "DWI"}
    assert len(case.organ_segs) == 1
    assert len(case.cancer_segs) == 3
    # Filename is authoritative: float tumor masks are still classified as cancer.
    assert all(s.volume_type == VolumeType.CANCER_SEG for s in case.cancer_segs)
    assert {s.reader for s in case.cancer_segs} == {"reader1", "reader2"}
    assert case.geometry_consistent


def test_flat_layout_subfolders_dont_collide(tmp_path: Path) -> None:
    """Same CaseNN in different sub-folders are different patients, not one merged case.

    PROMISE12 reuses ``Case00``..``Case29`` across ``TestData`` and ``TrainingData``.
    A recursive scan must namespace flat-layout ids by their sub-folder so the files
    stay separate (otherwise two patients merge into one geometrically inconsistent case).
    """
    test_dir = tmp_path / "TestData"
    train_dir = tmp_path / "TrainingDataPart1"
    test_dir.mkdir()
    train_dir.mkdir()
    # Same id, deliberately different geometry per sub-folder.
    _write_mri(test_dir / "Case00.mhd", size=(8, 8, 4))
    _write_mask(test_dir / "Case00_segmentation.mhd", size=(8, 8, 4))
    _write_mri(train_dir / "Case00.mhd", size=(16, 16, 4))
    _write_mask(train_dir / "Case00_segmentation.mhd", size=(16, 16, 4))

    ds = discover_cases(tmp_path, default_mri_type="T2W")
    assert {c.case_id for c in ds.cases} == {"TestData/Case00", "TrainingDataPart1/Case00"}
    for case in ds.cases:
        assert len(case.mris) == 1  # not merged: exactly one MRI each
        assert len(case.organ_segs) == 1
        assert case.geometry_consistent


def test_parent_of_patient_dirs(tmp_path: Path) -> None:
    """Pointing at a parent of patient dirs yields one case per dir."""
    for pid in ("020", "021"):
        d = tmp_path / pid
        d.mkdir()
        _write_mri(d / "t2.nii.gz")
        _write_mask(d / "t2_anatomy_reader1.nii.gz")
    ds = discover_cases(tmp_path)
    assert {c.case_id for c in ds.cases} == {"020", "021"}


def test_float_tumor_mask_not_reclassified(tmp_path: Path) -> None:
    patient = tmp_path / "021"
    patient.mkdir()
    _write_mask(patient / "t2_tumor_reader1.nii.gz", dtype=np.float32, label=3)
    vol = load_volume(patient / "t2_tumor_reader1.nii.gz")
    assert vol.volume_type == VolumeType.CANCER_SEG
    assert vol.num_labels == 2  # {0, 3}


def test_metadata_only_scan_skips_arrays(tmp_path: Path) -> None:
    _write_mri(tmp_path / "Case01.mhd")
    _write_mask(tmp_path / "Case01_segmentation.mhd")
    ds = discover_cases(tmp_path, load_arrays=False)
    case = ds.cases[0]
    assert case.primary_mri is not None and case.primary_mri.array is None
    assert case.organ_segs[0].num_labels is None  # needs pixel data


def test_geometry_mismatch_flagged(tmp_path: Path) -> None:
    patient = tmp_path / "021"
    patient.mkdir()
    _write_mri(patient / "t2.nii.gz", size=(8, 8, 4))
    _write_mask(patient / "t2_anatomy_reader1.nii.gz", size=(16, 16, 4))
    with pytest.warns(UserWarning, match="geometry"):
        ds = discover_cases(patient)
    assert ds.cases[0].geometry_consistent is False


def test_summary_renders(tmp_path: Path) -> None:
    patient = tmp_path / "021"
    patient.mkdir()
    _write_mri(patient / "t2.nii.gz")
    _write_mask(patient / "t2_anatomy_reader1.nii.gz")
    _write_mask(patient / "t2_tumor_reader1.nii.gz", dtype=np.float32, label=3)
    out = discover_cases(patient).summary()
    assert "021" in out
    assert "1 case(s); 1 with cancer segmentation" in out


# --------------------------------------------------------------------------- #
# Opt-in test against real data (set SHAPEMODELING_DATA_DIR)
# --------------------------------------------------------------------------- #

_REAL_DIR = os.environ.get("SHAPEMODELING_DATA_DIR")


@pytest.mark.skipif(not _REAL_DIR, reason="SHAPEMODELING_DATA_DIR not set")
def test_real_data() -> None:
    ds = discover_cases(_REAL_DIR, default_mri_type="T2W")
    assert len(ds) > 0
    for case in ds.cases:
        assert case.primary_mri is not None, f"{case.case_id} has no MRI"
        assert case.geometry_consistent, f"{case.case_id} geometry mismatch"
