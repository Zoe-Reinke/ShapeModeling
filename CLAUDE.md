# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Statistical Shape Model (SSM) pipeline for organ anatomy from MRI volumes and segmentation masks. Handles healthy and cancerous cases; cancer's correlation with shape modes is assessed statistically. The final model is exported for downstream use (e.g. segmentation, simulation, augmentation).

**Pipeline stages:**
1. Load MRI + segmentation masks (`src/shape_modeling/io/reader.py`) — done
2. Quality control of volumes + masks (`src/shape_modeling/qc.py`) — done
3. Extract surface meshes from masks (`src/shape_modeling/analysis/surface.py`) — first pass done
4. Align / establish correspondence across subjects (`src/shape_modeling/analysis/`) — first pass done
5. PCA-based SSM construction (`src/shape_modeling/analysis/ssm.py`) — first pass done
6. Statistical group analysis (healthy vs. cancerous)
7. Model export

Stages 3–5 currently exist as an **exploratory first pass** (see Analysis below): good enough
to inspect the cohort, not the final model.

## Layout & commands

- **Layout**: src-layout. Package at `src/shape_modeling/`; tests in `tests/`.
- **Install**: `pip install -e ".[dev]"`; add `.[analysis]` for the analysis stage
  (scikit-image / scikit-learn / scipy). Use a venv; `.venv/` is gitignored.
- **Test**: `pytest` / single test `pytest tests/test_reader.py::test_name`.
- **Lint / types**: `ruff check src tests` · `mypy src`.

## Data

Input volumes are **MetaImage (`.mhd`/`.raw`)** or **NIfTI (`.nii`/`.nii.gz`)**, loaded
via **SimpleITK** (the only reader that handles both). Two layouts are supported:

- **Per-patient directory** (`Prostate158/`): one folder per case (`020/`, `021/`, …, 139
  cases), each with MRI sequences (`t2`, `adc`, `dwi`) and segmentations named
  `<sequence>_<structure>_reader<N>.nii.gz`.
- **Flat** (`PROMISE12/`): cases keyed by filename (`CaseXX.mhd` + `CaseXX_segmentation.mhd`).
  PROMISE12 splits these across sub-folders (`TestData/`, `TrainingDataPart1..3/`) that
  **reuse the same `CaseNN` ids for different patients** — so flat ids are namespaced by
  sub-folder (see below).

Reader (`src/shape_modeling/io/reader.py`) — `discover_cases(dir)` returns a `Dataset` of
`Case` objects. A `Case` holds `mris: dict[sequence -> LoadedVolume]`, `organ_segs: list`,
and `cancer_segs: list` (one entry per reader annotation), plus `primary_mri` (prefers
T2). Key behaviours:

- **Filename tokens are authoritative over pixel dtype.** `tumor`/`lesion` → cancer,
  `anatomy`/`segmentation` → organ, `reader` → a segmentation, bare sequence name → MRI.
  Tumor masks are sometimes stored as `float` but are still binary label maps — never
  classify by dtype. Tokens live in `CANCER_TOKENS`/`ORGAN_TOKENS`/`SEQUENCE_TOKENS`.
- **Layout is auto-detected per directory**: if any file is a bare sequence name (case id
  parses to `""`), the whole directory is one case named after the folder; otherwise the
  filename carries the case id (flat).
- **Flat ids are namespaced by sub-folder** relative to the scan root, so the same `CaseNN`
  in `TestData/` vs `TrainingDataPart1/` stay distinct (`TestData/Case00` ≠
  `TrainingDataPart1/Case00`). A plain flat folder (files directly under the scan root) keeps
  bare ids; per-patient dirs keep the folder name. Without this, recursive discovery merged
  different patients into one geometry-inconsistent case.
- `IGNORE_TOKENS` (`empty`) skips all-zero placeholder files (e.g. `022/empty.nii.gz`,
  meaning "no tumor").
- MRI **sequence type is not in the headers** — inferred from filename, else
  `default_mri_type`, else `None`. Geometry mismatches vs the reference MRI are **flagged
  (`geometry_consistent`), not resampled** (resampling is a later stage).

Opt-in real-data test: `SHAPEMODELING_DATA_DIR=<patient-dir> pytest -k real`.
Manual check: `notebooks/test_reader.ipynb` (set `DATA_DIR`).

## Quality control

`src/shape_modeling/qc.py` — `run_qc(dataset, config=QCConfig()) -> QCReport`. Inspects
every volume/mask and returns `QCFinding`s tagged INFO/WARNING/ERROR; **non-destructive
and never raises** (advisory — caller decides what to exclude). `QCReport.summary()` prints
a per-case table (`worst` = `clean` when no findings) plus warning/error detail;
`report.passed` is true when there are no ERRORs.

Three check groups: **image integrity** (finite values, positive spacing, anisotropy,
non-constant MRI, geometry consistency), **mask validity** (non-empty, label sanity,
connected components / islands, FOV-border contact, physical volume), **cross-mask**
(tumor inside organ, mask over tissue not background). Thresholds live in `QCConfig` and
default to **prostate-plausible** values (e.g. `organ_volume_range_cm3=(10, 200)`) — pass a
different config for thyroid/carotid. Reuses `reader.geometry_matches`. SimpleITK+numpy
only. Note: SimpleITK's NIfTI writer drops NaN, so the finite check is unit-tested on an
in-memory array, not a written fixture.

## Analysis (exploratory SSM, stages 3–5)

`src/shape_modeling/analysis/` turns a `Dataset` into a first-pass shape model so the cohort
can be inspected before the final model. Driver: `notebooks/explore_shapes.ipynb`. This is
**throwaway-quality** — the correspondence is "good enough to look at", not final.

- **`cohort.py`** — `build_cohort({source_label: Dataset}) -> list[CohortEntry]`. Keeps
  **healthy** cases (`is_healthy` = no `cancer_segs`) that have an organ seg; each entry
  carries `source`, `volume_cm3`, and the chosen organ `LoadedVolume`. Whole gland =
  `organ_segs[0] > 0` (first reader; binarised for cross-dataset comparability). Discover each
  dataset separately so the source label is unambiguous. Healthy cohort on current data ≈ 136
  (80 PROMISE12 + 56 Prostate158; the other 83 Prostate158 cases have tumour segs).
- **`surface.py`** — `clean_mask` (largest connected component, fill holes, optional Gaussian
  anti-alias) then `extract_surface` (marching cubes → mesh in **physical mm** via the
  volume's SimpleITK index→physical transform). `save_ply` writes a mesh for viewing.
  Surfaces are for inspection; the correspondence step feeds segmentations to ShapeWorks.
  `hull_surface(points)` triangulates a corresponded particle cloud (convex hull — particles
  carry no connectivity) and `export_mode_shapes(model, out_dir, ...)` writes the mean walked
  ±sd along each mode to PLYs (reusing the **mean's** hull faces so vertex *j* is consistent)
  for viewing in 3D Slicer / ParaView. The convex hull is a quick blob only — it keeps just the
  extreme particles and bridges concavities; for a **faithful** mean/mode mesh use
  `warp_surface(template, src_points, dst_points)`: thin-plate-spline (scipy RBF) warp of a
  *real* groomed surface so its particles land on the target, inheriting the template's faces.
  `src_points` and the template verts must share a frame — pair a subject's groomed-DT marching-
  cubes surface with its `*_local.particles` (local sits in the DT frame; `*_world.particles` are
  Procrustes-rotated, the frame `model.mean` is built in). See the mean-shape cell in
  `explore_shapes.ipynb`.
- **`correspondence.py`** — ShapeWorks particle correspondence. **ShapeWorks is NOT
  pip-installable**; install it separately and run from its environment (`import shapeworks`
  + `shapeworks` CLI on PATH). The import is deferred to call time, so the rest of the package
  works without it. `run_shapeworks(cohort, ShapeWorksConfig) -> ParticleSystem` (caches
  particles in `work_dir/particles.npz`). `load_particles(dir, case_ids)` reads particle files
  from any ShapeWorks run (e.g. Studio) with no package needed. Case ids are sanitised
  (`/`→`__`) for filenames. **Grooming** (`_groom`): pad(0) → antialias → resample(Linear) →
  `binarize()` → `computeDT(0.5)`. Order matters — padding *after* antialias coats the DT border
  in the surface value, and skipping the re-`binarize()` leaves the resample-smeared interface in
  the DT; either makes the transform degenerate and scatters particles (see [[grooming-dt-bug]]).
  Both fresh and cached particles are gated by `particle_spread_ratio` (median over subjects of
  particle-extent / organ-extent; healthy ≈ 1.0–1.3, degenerate ≈ 3+); `run_shapeworks` raises if
  it exceeds `ShapeWorksConfig.max_spread_ratio` (default 1.75; `None` disables), so a stale or
  degenerate cache is rejected, not silently analysed.
- **`ssm.py`** — `gpa` (generalized Procrustes; removes translation/rotation/optional scale)
  then `build_ssm` (PCA → `ShapeModel` with mean, modes, eigenvalues, scores;
  `n_components` capped at `min(S-1, 3M)`). `mode_shape(model, mode, sd)` walks the mean along
  a mode for visualisation. `centroid_size(shape)` is Procrustes centroid size (scalar per
  shape); since `gpa(scale=True)` normalises every shape to unit Frobenius norm, **aligned
  coordinates are unitless (~0.0x, not mm)** and aligned centroid size is ~constant — to test
  whether a mode is a *size* mode, correlate its score against centroid size of the **raw**
  (pre-`gpa`) particles, not the aligned ones. numpy + scipy + scikit-learn.

Synthetic tests in `tests/test_analysis.py` (no ShapeWorks needed); the ShapeWorks optimiser
itself is an opt-in smoke test (`pytest.importorskip("shapeworks")`).
