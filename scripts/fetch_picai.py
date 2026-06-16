#!/usr/bin/env python3
"""Fetch a subset of PI-CAI whole-gland AI delineations + their cancer labels.

PI-CAI's AI-generated whole-gland masks live in the public ``picai_labels`` repo
(``anatomical_delineations/whole_gland/AI/<subfolder>/``) as ``{patient}_{study}.nii.gz``
NIfTI label maps. The MRI volumes themselves are *not* needed here -- the shape pipeline
builds from the segmentation mask alone (``build_grouped_cohort`` only reads the organ
mask array).

This downloader pulls masks and renames each
``{patient}_{study}_segmentation.nii.gz`` so ``shape_modeling.io.reader.discover_cases``
classifies it as an ORGAN_SEG (the ``segmentation`` token) and parses the case id as
``{patient}_{study}``. The AI folder carries no cancer status, so we also download the
repo's ``clinical_information/marksheet.csv`` and emit a ``labels.csv`` sidecar mapping
each fetched case to its ``case_csPCa`` flag, acquisition ``center``, and clinical
``prostate_volume`` -- the notebook reads it to set ``CohortEntry.is_cancer``.

The script is **additive and idempotent**: cases already present in ``--out`` are never
re-downloaded, and ``labels.csv`` is regenerated from everything on disk after each run.
Two selection modes:

    # first-N unused cases (any cancer status)
    python scripts/fetch_picai.py --n 50 --out PICAI

    # stratified: N new csPCa-positive + N new csPCa-negative cases not already fetched
    python scripts/fetch_picai.py --n-cancer 35 --n-benign 15 --out PICAI

Stdlib only (urllib/csv/json), so no extra deps.

Caveat: these gland masks are *AI*-derived. Pooling them with the human-annotated
Prostate158 / SAML masks introduces a domain shift, not just biology -- treat the
resulting model comparison as a sensitivity check, not a validated finding.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "DIAGNijmegen/picai_labels"
BRANCH = "main"
# whole_gland/AI has two label sources (Bosma22b, Guerbet23); Bosma22b is the nnU-Net set
# used across the PI-CAI public training cohort.
AI_SUBDIR = "anatomical_delineations/whole_gland/AI"
MARKSHEET_PATH = "clinical_information/marksheet.csv"
SEG_SUFFIX = "_segmentation.nii.gz"  # how a fetched mask is named on disk

_API = "https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
_RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def _get(url: str) -> bytes:
    """GET a URL with a User-Agent (GitHub rejects API calls without one)."""
    req = urllib.request.Request(url, headers={"User-Agent": "shape-modeling-fetch-picai"})
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted GitHub hosts)
            return resp.read()
    except urllib.error.HTTPError as exc:  # surface rate limits / 404s clearly
        body = exc.read().decode("utf-8", "replace")[:300]
        raise SystemExit(f"HTTP {exc.code} fetching {url}\n{body}") from exc


def list_masks(subfolder: str) -> list[dict]:
    """Return the GitHub contents listing for the AI mask subfolder (sorted by name)."""
    url = _API.format(repo=REPO, path=f"{AI_SUBDIR}/{subfolder}", branch=BRANCH)
    entries = json.loads(_get(url))
    if not isinstance(entries, list):  # a dir with >1000 files still returns a list (first page)
        raise SystemExit(f"unexpected API response for {url}: {entries!r:.200}")
    masks = [e for e in entries if e["type"] == "file" and e["name"].endswith(".nii.gz")]
    masks.sort(key=lambda e: e["name"])
    return masks


def load_marksheet() -> dict[str, dict]:
    """Map ``{patient}_{study}`` -> marksheet row (csPCa / center / prostate_volume)."""
    raw = _get(_RAW.format(repo=REPO, branch=BRANCH, path=MARKSHEET_PATH))
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    out: dict[str, dict] = {}
    for row in reader:
        case_id = f"{row['patient_id']}_{row['study_id']}"
        out[case_id] = row
    return out


def _is_cancer(stem: str, marksheet: dict[str, dict]) -> bool:
    row = marksheet.get(stem)
    return row is not None and row.get("case_csPCa", "").strip().upper() == "YES"


def select_new(
    masks: list[dict],
    existing: set[str],
    marksheet: dict[str, dict],
    *,
    n: int,
    n_cancer: int | None,
    n_benign: int | None,
) -> list[dict]:
    """Pick mask entries to download, skipping anything already on disk.

    Stratified mode (``n_cancer``/``n_benign`` set): walk the sorted masks and take the next
    unused csPCa-positive / -negative cases until each quota is met (cases missing from the
    marksheet can't be classified, so they're skipped). Otherwise: the next ``n`` unused cases.
    """
    fresh = [e for e in masks if e["name"][: -len(".nii.gz")] not in existing]
    if n_cancer is None and n_benign is None:
        return fresh[:n]

    want_c, want_b = n_cancer or 0, n_benign or 0
    picked: list[dict] = []
    got_c = got_b = 0
    for e in fresh:
        stem = e["name"][: -len(".nii.gz")]
        if stem not in marksheet:  # can't classify by cancer status -> skip in stratified mode
            continue
        if _is_cancer(stem, marksheet):
            if got_c < want_c:
                picked.append(e)
                got_c += 1
        elif got_b < want_b:
            picked.append(e)
            got_b += 1
        if got_c >= want_c and got_b >= want_b:
            break
    if got_c < want_c or got_b < want_b:
        print(f"  WARNING: only found {got_c}/{want_c} cancer and {got_b}/{want_b} benign "
              f"unused cases in the listing")
    print(f"  selecting {len(picked)} NEW cases ({got_c} cancer, {got_b} benign)")
    return picked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=50, help="first-N unused cases (default 50)")
    ap.add_argument("--n-cancer", type=int, default=None, help="NEW csPCa-positive cases to fetch")
    ap.add_argument("--n-benign", type=int, default=None, help="NEW csPCa-negative cases to fetch")
    ap.add_argument("--out", type=Path, default=Path("PICAI"), help="output dir (default PICAI/)")
    ap.add_argument("--subfolder", default="Bosma22b", help="AI label source under whole_gland/AI")
    args = ap.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("downloading marksheet ...")
    marksheet = load_marksheet()

    print(f"listing {AI_SUBDIR}/{args.subfolder} ...")
    masks = list_masks(args.subfolder)
    existing = {p.name[: -len(SEG_SUFFIX)] for p in out.glob(f"*{SEG_SUFFIX}")}
    if existing:
        print(f"  {len(existing)} cases already present -> will not re-download")

    picked = select_new(
        masks, existing, marksheet, n=args.n, n_cancer=args.n_cancer, n_benign=args.n_benign
    )
    for i, entry in enumerate(picked, 1):
        stem = entry["name"][: -len(".nii.gz")]
        dest = out / f"{stem}{SEG_SUFFIX}"
        print(f"  [{i:>3}/{len(picked)}] {entry['name']} -> {dest.name}")
        dest.write_bytes(_get(entry["download_url"]))

    # Regenerate labels.csv from EVERY mask on disk (existing + newly fetched), so the sidecar
    # always reflects the full cohort regardless of how many runs built it.
    present = sorted({p.name[: -len(SEG_SUFFIX)] for p in out.glob(f"*{SEG_SUFFIX}")})
    label_rows: list[dict] = []
    missing_labels: list[str] = []
    for stem in present:
        row = marksheet.get(stem)
        if row is None:
            missing_labels.append(stem)
            label_rows.append(
                {"case_id": stem, "is_cancer": "", "center": "", "prostate_volume": ""}
            )
        else:
            label_rows.append(
                {
                    "case_id": stem,
                    "is_cancer": "1" if _is_cancer(stem, marksheet) else "0",
                    "center": row.get("center", ""),
                    "prostate_volume": row.get("prostate_volume", ""),
                }
            )

    labels_path = out / "labels.csv"
    with labels_path.open("w", newline="") as fh:
        fields = ["case_id", "is_cancer", "center", "prostate_volume"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(label_rows)

    n_cancer = sum(r["is_cancer"] == "1" for r in label_rows)
    n_benign = sum(r["is_cancer"] == "0" for r in label_rows)
    print(f"\ndownloaded {len(picked)} new masks; {len(present)} total in {out}/")
    print(f"  cohort now: {n_cancer} cancer, {n_benign} benign  (-> {labels_path})")
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} cases not in marksheet (labels blank): "
              f"{missing_labels[:5]}{' ...' if len(missing_labels) > 5 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
