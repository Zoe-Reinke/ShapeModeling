# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Statistical Shape Model (SSM) pipeline for organ anatomy from MRI volumes and segmentation masks. Handles healthy and cancerous cases; cancer's correlation with shape modes is assessed statistically. The final model is exported for downstream use (e.g. segmentation, simulation, augmentation).

**Pipeline stages:**
1. Load MRI + segmentation masks (`src/shape_modeling/io/reader.py`) — done
2. Quality control of volumes + masks (`src/shape_modeling/qc.py`) — done
3. Extract surface meshes from masks (`src/shape_modeling/analysis/surface.py`) — first pass done
4. Align / establish correspondence across subjects (`src/shape_modeling/analysis/`) — first pass done
   (multi-center cohorts also **ComBat-harmonized on site** between GPA and PCA, `analysis/harmonize.py`)
5. PCA-based SSM construction (`src/shape_modeling/analysis/ssm.py`) — first pass done
6. Statistical group analysis (healthy vs. cancerous) (`src/shape_modeling/analysis/groups.py`) — first pass done
7. Model export

Stages 3–6 currently exist as an **exploratory first pass** (see Analysis below): good enough
to inspect the cohort and answer "is there a signal?", not the final model. The correspondence
is throwaway-quality, so any group effect is *suggestive*, not a validated biomarker.

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
- **Flat** (`PROMISE12/`, `SAML/`): cases keyed by filename (`CaseXX.mhd` + `CaseXX_segmentation.mhd`).
  PROMISE12 splits these across sub-folders (`TestData/`, `TrainingDataPart1..3/`) that
  **reuse the same `CaseNN` ids for different patients** — so flat ids are namespaced by
  sub-folder (see below). `SAML/` is the multi-center prostate set used in place of PROMISE12
  for the analysis (one sub-folder per acquisition site: `BIDMC/`, `BMC/`, `HK/`, `I2CVB/`,
  `RUNMC/`, `UCL/`), so each site becomes its own labelled source.

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

## Analysis (exploratory SSM, stages 3–6)

`src/shape_modeling/analysis/` turns a `Dataset` into a first-pass shape model so the cohort
can be inspected before the final model. Three notebook drivers, all **throwaway-quality** (the
correspondence is "good enough to look at", not final):
- `notebooks/explore_shapes.ipynb` — healthy whole-gland SSM, multi-center (Prostate158 + SAML),
  with the full site/acquisition/annotation confound tests (section 6). The canonical `model` is
  **ComBat-harmonized on centre** (built in section 4); the un-harmonized fit is kept as `model_raw`
  for the section-6 before/after evidence (see [[exploratory-ssm-findings]]).
- `notebooks/explore_shapes_alldata.ipynb` — pooled **anatomical** basis over *all* glands
  (healthy + cancer, multi-center) via `build_grouped_cohort`, also ComBat-harmonized on centre
  (cancer is within the Prostate158 batch, so the contrast is preserved). Descriptive only — **not**
  a substrate for the cancer question (cancer is confounded with source there; do that test in the
  Prostate158-only grouped run).
- `notebooks/explore_cancer_shape.ipynb` — stage-6 headline test, **within Prostate158** only.

- **`cohort.py`** — `build_cohort({source_label: Dataset}) -> list[CohortEntry]` keeps **healthy**
  cases (`is_healthy` = no `cancer_segs`) with an organ seg; `build_grouped_cohort(...)` keeps
  **every** case with an organ seg and tags each `is_cancer`. Each entry carries `source`,
  `volume_cm3`, `is_cancer`, and the chosen organ `LoadedVolume`. Whole gland = `organ_segs[0] > 0`
  (first reader; binarised for cross-dataset comparability). Discover each dataset separately so
  the source label is unambiguous. Current healthy cohort = 153 (56 Prostate158 + 97 SAML, after
  excluding the `I2CVB` site whose masks are planar-truncated — see [[exploratory-ssm-findings]]);
  the all-data basis adds the 83 Prostate158 cancer cases on top.
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
  (`/`→`__`) for filenames. **Grooming** (`_groom_cohort`, the whole cohort together) follows the
  ShapeWorks recommended segmentation workflow so the groomed DTs are **pose-normalised** (Studio
  shows them aligned — the optimizer's own `procrustes` only aligns *particles*, not the groomed
  images): per subject `antialias → resample(Linear, iso) → binarize() → centre-of-mass align`;
  then pick a medoid reference (`find_reference_image_index`) and rigidly ICP-align every shape to
  it (`createRigidRegistrationTransform` **requires DTs**, so per-subject DTs are built first, used
  to choose+align to the reference, then the transform is applied to the binary); then per subject
  `fitRegion(own bbox + pad) → computeDT(0.5) → gaussianBlur(blur_sigma)`. Three things bite:
  re-`binarize()` after the Linear resample (it smears the 0-crossing into a band; `computeDT(0.5)`
  needs a crisp 0/1 mask — see [[grooming-dt-bug]]); apply the rigid transform **resampled onto the
  reference's grid** (`applyTransform(t, ref.origin/dims/spacing/coordsys, …)`), **not in place** —
  an in-place transform keeps the subject's own grid, so a rotation/translation that swings the
  gland past it clips the gland away (case `054` came out *empty*; no fixed pad reliably absorbs
  every subject's rotation, esp. grids whose origin sits far from `(0,0,0)`). Resampling onto the
  (pad-grown) reference grid lands every aligned gland inside that frame by construction. `_groom_cohort`
  also raises a **named** error if any subject's mask is empty after step 1 (eroded by antialias) or
  step 2 (fell outside the frame). And use `fitRegion`, **not `crop`**, around the step-3 bbox: when
  a gland sits within `pad` of a face, the padded bbox spills past the grid and `crop` raises
  `ValueError: Invalid region … outside physical bounds`; `fitRegion` pads with background instead, so
  the DT keeps a margin on every face. The `*_local.particles` now sit in the **aligned+cropped** DT
  frame, so `warp_surface` pairing (subject groomed-DT surface ↔ its local particles) stays self-consistent.
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
- **`harmonize.py`** — `combat_harmonize(particles, batch, continuous=None, …)` removes acquisition
  **site** effects from corresponded particle coordinates with ComBat (empirical-Bayes location/scale),
  so a multi-center basis encodes anatomy not scanner. Inserts **between GPA and PCA**: harmonize the
  `(S, M, 3)` GPA-aligned coords against `batch` (centre), protecting raw centroid size as a
  `continuous` covariate so real size variation survives, then re-`gpa` + `build_ssm`. ComBat needs
  *corresponded* points (it corrects feature-by-feature), so it runs **after** ShapeWorks, never before.
  Uses **neuroCombat** (its API is features×samples — the helper transposes; size goes in
  `continuous_cols`, not as a batch). neuroCombat is optional (in the `.[analysis]` extra, with
  `pandas`) and the import is **deferred to call time** like ShapeWorks. Verify on the re-fit scores:
  Kruskal site p should drop out of significance while mode-1 `r(raw centroid size)` survives (~0.36);
  if size collapses, it got entangled with site. Smallest sites have n=12, so the EB scale terms stay
  noisy even after shrinkage — expected.
- **`groups.py`** (stage 6, healthy vs. cancer) — `compare_groups(model, cohort)` bundles:
  `per_mode_tests` (per-mode Mann-Whitney + Cohen's d, BH/FDR-corrected), `shape_predicts_cancer`
  (omnibus "does shape predict cancer?" — CV logistic AUC + label-permutation p over the top
  modes), `group_mean_difference_shape` (per-point mm displacement between the two group-mean
  glands), and `volume_difference` (raw size gap). Driver: `explore_cancer_shape.ipynb`, run
  **within Prostate158** with GPA `scale=False` (keeps size; volume gap reported separately).
  See [[stage6-cancer-shape-analysis.md]] for the current result.

Synthetic tests in `tests/test_analysis.py` (no ShapeWorks needed); the ShapeWorks optimiser
itself is an opt-in smoke test (`pytest.importorskip("shapeworks")`).
