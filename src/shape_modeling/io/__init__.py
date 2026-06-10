"""I/O for medical image volumes and segmentation masks."""

from shape_modeling.io.reader import (
    Case,
    Dataset,
    LoadedVolume,
    VolumeType,
    classify_file,
    detect_reader,
    detect_sequence,
    discover_cases,
    load_volume,
    parse_case_id,
)

__all__ = [
    "Case",
    "Dataset",
    "LoadedVolume",
    "VolumeType",
    "classify_file",
    "detect_reader",
    "detect_sequence",
    "discover_cases",
    "load_volume",
    "parse_case_id",
]
