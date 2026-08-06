#!/usr/bin/env python3
"""
Prueft einen TotalSegmentator-Datensatz und entfernt alle Patienten,
die KEIN nicht-leeres thyroid_gland- UND trachea-Label besitzen.

Standard: Dry-Run (loescht nichts, zeigt nur an).
  --delete            : loescht die Ordner tatsaechlich (unumkehrbar!)
  --quarantine ORDNER : verschiebt sie dorthin, statt zu loeschen (sicherer)

Unterstuetzt zwei Layouts:
  1) Standard Zenodo: <subject>/segmentations/thyroid_gland.nii.gz  (+ trachea.nii.gz)
  2) Multilabel:      <subject>/<...>.nii.gz mit Labels 17 (thyroid) und 16 (trachea)

Beispiele:
  python prune_totalsegmentator_thyroid.py /pfad/Totalsegmentator_dataset_v201
  python prune_totalsegmentator_thyroid.py /pfad/dataset --quarantine /pfad/verworfen
  python prune_totalsegmentator_thyroid.py /pfad/dataset --delete
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
import os

# TotalSegmentator v2 Label-IDs (nur fuer Multilabel-Layout relevant)
THYROID_ID = 17
TRACHEA_ID = 16


def nonempty(path: Path) -> bool:
    """True, wenn die Maske existiert und mindestens ein Voxel > 0 enthaelt."""
    if not path.is_file():
        return False
    # nibabel erzwingt KEINE orthonormalen Richtungskosinus -> liest die
    # TotalSegmentator-Header, an denen ITK/SimpleITK scheitert.
    arr = np.asanyarray(nib.load(str(path)).dataobj)
    return bool(np.any(arr > 0))


def label_present(path: Path, label_id: int) -> bool:
    """True, wenn im Multilabel-Volume das gegebene Label vorkommt."""
    if not path.is_file():
        return False
    arr = np.asanyarray(nib.load(str(path)).dataobj)
    return bool(np.any(arr == label_id))


def has_both_labels(subject: Path) -> bool:
    """Prueft beide Layouts. Behalten = thyroid UND trachea nicht leer."""
    seg_dir = subject / "segmentations"
    if seg_dir.is_dir():  # Layout 1: eine Datei pro Klasse
        return nonempty(seg_dir / "thyroid_gland.nii.gz") and \
               nonempty(seg_dir / "trachea.nii.gz")

    # Layout 2: eine Multilabel-Datei im Subject-Ordner
    for ml in list(subject.glob("*.nii.gz")) + list(subject.glob("*.nii")):
        if ml.name.startswith("ct"):        # das CT selbst ueberspringen
            continue
        if label_present(ml, THYROID_ID) and label_present(ml, TRACHEA_ID):
            return True
    return False


def is_subject_dir(p: Path) -> bool:
    """Heuristik: Ordner mit ct.nii.gz oder segmentations/ gilt als Patient."""
    return p.is_dir() and (
        (p / "segmentations").is_dir() or any(p.glob("ct.nii*"))
    )

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Wurzelordner des Datensatzes")
    ap.add_argument("--delete", action="store_true",
                    help="Ordner wirklich loeschen (unumkehrbar)")
    ap.add_argument("--quarantine", type=Path, default=None,
                    help="Verzeichnis, in das verworfene Faelle verschoben werden")
    args = ap.parse_args()

    subjects = sorted(p for p in args.root.iterdir() if is_subject_dir(p))
    if not subjects:
        print(f"Keine Patientenordner in {args.root} gefunden. "
              f"Zeigt der Pfad auf die Datensatz-Wurzel?")
        return

    keep, drop = [], []
    for s in subjects:
        try:
            (keep if has_both_labels(s) else drop).append(s)
        except Exception as e:  # kaputte/unlesbare Datei -> zum Verwerfen markieren
            print(f"  [WARN] {s.name}: nicht lesbar ({e}) -> wird verworfen")
            drop.append(s)

    print(f"\nGeprueft:  {len(subjects)} Patienten")
    print(f"Behalten:  {len(keep)}  (thyroid + trachea vorhanden)")
    print(f"Verwerfen: {len(drop)}\n")

    for k in keep:
        seg_dir = k / "segmentations"
        if seg_dir.is_dir():  # Layout 1: eine Datei pro Klasse
            for k in seg_dir.iterdir():
                if k.name not in ("thyroid_gland.nii.gz", "trachea.nii.gz"):
                    print(f"  [WARN] {k.name} in {seg_dir} wird verworfen")
                    if args.delete:
                        os.remove(k)
                    else:
                        print(f"    (Dry-Run: nicht geloescht)")


    if not drop:
        print("Nichts zu tun.")
        return

    if args.quarantine:
        args.quarantine.mkdir(parents=True, exist_ok=True)
        for s in drop:
            print(f"  verschiebe {s.name} -> {args.quarantine}")
            shutil.move(str(s), str(args.quarantine / s.name))
        print(f"\n{len(drop)} Faelle nach {args.quarantine} verschoben.")
    elif args.delete:
        for s in drop:
            print(f"  loesche {s.name}")
            shutil.rmtree(s)
        print(f"\n{len(drop)} Faelle geloescht.")
    else:
        print("DRY-RUN — es wurde nichts veraendert. "
              "Wuerde folgende Faelle entfernen:")
        for s in drop:
            print(f"  {s.name}")
        print("\nZum Ausfuehren: --quarantine <ordner>  (sicher)  oder  --delete")


if __name__ == "__main__":
    main()
