# ShapeModeling

A Statistical Shape Model (SSM) pipeline for organ anatomy, built from medical-image segmentation masks. The pipeline turns a cohort of binary organ masks into a compact
linear model of shape variation — a **mean shape** plus a set of orthogonal **deformation
modes** — and provides the statistics to ask whether a per-subject covariate (e.g. a disease
label, a site, an acquisition protocol) is associated with shape. The fitted model is exported
in a self-contained form for downstream use: segmentation priors, simulation, augmentation, or
fitting to a new modality.

The code is **organ-agnostic**: nothing in the readers, QC, surface extraction, correspondence,
alignment, PCA, or group analysis is specialised to a particular anatomy. Anatomy-specific
choices live entirely in *configuration* (QC plausibility thresholds, particle counts) and in
the *labels* you attach to each case.

## What an SSM is, concretely

Every subject is reduced to the same number of points **M**, in the same order, so that point
*j* marks the same anatomical location on every subject — this is *correspondence*. Stacking
**S** subjects gives a tensor of shape `(S, M, 3)`. The model is then:

```
x(b) = mean + Σ_k  b_k · √λ_k · φ_k
```

where `mean` is the average shape `(M, 3)`, `φ_k` are the principal modes (unit eigenvectors of
the shape covariance), `λ_k` their variances, and `b_k` are **dimensionless weights in units of
standard deviation** (`b_k = 0` is the mean, `|b_k| ≤ 3` spans the plausible cohort range).
Those weights are the only knobs a downstream optimiser turns.

## Pipeline

| Stage | What it does | Module |
|-------|--------------|--------|
| 1 | **Load** image volumes + segmentation masks | `io/reader.py` |
| 2 | **Quality control** of volumes + masks | `qc.py` |
| 3 | **Surface extraction** from masks | `analysis/surface.py` |
| 4 | **Correspondence** — place M matched particles per shape | `analysis/correspondence.py` |
| 5 | **Alignment + SSM** — Procrustes then PCA | `analysis/ssm.py` |
| 6 | **Group analysis** — does a covariate track shape? | `analysis/groups.py` |
| 7 | **Model export** for downstream fitting | `analysis/export.py` |

Optional cross-stage step: **batch-effect harmonization** (`analysis/harmonize.py`) between
stages 5 and 6 for multi-source cohorts. SSM quality is judged with the standard
compactness / generalization / specificity curves (`analysis/evaluation.py`).

The **library is code-complete through stage 7** — every stage is an implemented, unit-tested
module. What separates "runs" from "final" is *running the production path*, not more code: the
exploratory **notebooks** use a throwaway-quality correspondence (good enough to inspect a cohort
and answer "is there a signal?"), so any group effect from them is *suggestive*, not a biomarker.
The final model is the one [`scripts/build_prostate_ssm.py`](scripts/build_prostate_ssm.py) builds
at the full correspondence, validated on a held-out set by
[`scripts/validate_ssm.py`](scripts/validate_ssm.py) — see [Scripts](#scripts-production-path).

## Algorithms, stage by stage

### 3. Surface extraction (`surface.py`)

A raw label mask is rarely watertight, so before meshing:

- **`clean_mask`** binarises (`> 0`), keeps the **largest connected component** (drops islands
  via `scipy.ndimage.label` + voxel counts), **fills interior holes** (`binary_fill_holes`), and
  optionally Gaussian anti-aliases then re-thresholds at 0.5.
- **`extract_surface`** runs **marching cubes** (`skimage.measure`) at level 0.5, then maps the
  vertices from voxel `(z, y, x)` indices into **physical millimetres** `(x, y, z)` using the
  volume's full index→physical transform (spacing, origin, **and direction cosines**), so meshes
  from different scans share one world frame.

Two ways to mesh a *corresponded particle cloud* (particles carry no triangle connectivity):

- **`hull_surface`** — convex-hull triangulation. A quick blob: it keeps only the extreme points
  and bridges concavities. Fine for a roughly-convex organ; the hull faces of the **mean** can be
  reused for every shape since particle ordering is shared.
- **`warp_surface`** — the faithful route. Borrow connectivity from a *real* template mesh and
  carry it onto a target particle configuration via a **two-stage map**: a global **Umeyama
  similarity** (uniform scale + rotation + translation) absorbs the bulk frame/scale change, then
  a **thin-plate-spline** (`scipy` RBF) models only the small residual. Splitting it this way is
  what keeps the result from folding — a single spline asked to do the scale change as well
  oscillates between control points. Faces are inherited unchanged, so concavities and full
  resolution survive.
- **`smooth_surface`** — **Taubin (λ|μ)** smoothing on a uniform-weight graph Laplacian built from
  the triangle edges. Alternates a positive Laplacian step (`λ`, smooth + shrink) with a negative
  one (`μ`, slightly larger magnitude, smooth + re-inflate) so volume is retained, unlike plain
  Laplacian smoothing which collapses toward the centroid.
- **`export_mode_shapes`** writes the mean walked ±sd along each mode to PLY for inspection in
  3D Slicer / ParaView (reusing the mean's faces so vertex *j* is consistent across files).

### 4. Correspondence (`correspondence.py`)

Particle correspondence via **ShapeWorks** (not pip-installable; install separately, run from its
environment — the `import shapeworks` is deferred to call time so the rest of the package loads
without it). Per-subject **grooming** follows the ShapeWorks recommended segmentation workflow so
the groomed distance transforms are pose-normalised:

1. `antialias → resample(Linear, isotropic) → binarize()` (re-binarize after the linear resample —
   it smears the 0-crossing into a band, and the later `computeDT(0.5)` needs a crisp 0/1 mask).
2. **centre-of-mass align**, pick a **medoid reference**, build per-subject DTs, **ICP**-align
   every shape to the reference, and apply the rigid transform **resampled onto the (padded)
   reference grid** — not in place, or a rotation that swings the organ past its own grid clips it.
3. `fitRegion(own bbox + pad) → computeDT(0.5) → gaussianBlur` — `fitRegion`, not `crop`, so a
   padded bbox that spills past the grid pads with background instead of raising.

Then the optimizer places **M** particles (default 512) over the groomed shapes, and the
per-subject `*_world.particles` are read into an `(S, M, 3)` array. Results are **cached** in
`work_dir` and gated by `particle_spread_ratio` (median over subjects of
particle-extent / organ-extent; ~1.0–1.3 healthy, 3+ degenerate) so a stale or broken cache is
rejected rather than silently analysed.

### 5. Alignment + SSM (`ssm.py`)

- **`gpa`** — **Generalized Procrustes Alignment**. Removes per-shape translation, rotation, and
  (optionally) scale by iteratively aligning every shape to the evolving mean. Rotation per shape
  is the optimal proper rotation via **Kabsch/SVD** (`det = +1` enforced); the mean is re-centred
  (and re-normalised when `scale=True`) each pass until it converges. With `scale=True` every shape
  is normalised to **unit Frobenius norm**, so aligned coordinates are *unitless* — to test whether
  a mode is a size mode, correlate its scores against `centroid_size` of the **raw** (pre-GPA)
  points. Use `scale=False` to keep size when it is part of the question.
- **`build_ssm`** — PCA over the flattened `(S, 3M)` aligned coordinates. Returns a `ShapeModel`
  with the mean, components, eigenvalues, explained-variance ratios, and per-subject scores.
  `n_components` is capped at `min(S − 1, 3M)` — the most the data can support.
- **`mode_shape`** walks the mean `sd` standard deviations along a mode (`mean + sd·√λ·φ`).

### 5b. Harmonization (`harmonize.py`, optional)

When a cohort pools multiple acquisition sources, the leading PCA modes can encode the **scanner**
(coil, vendor, field strength, resolution) rather than anatomy. **`combat_harmonize`** applies
**ComBat** (Johnson et al. 2007, via `neuroCombat`): a per-batch **location/scale** model fit by
**empirical Bayes**, shrinking each batch's additive shift and multiplicative scale toward the
pooled estimate — which matters when the smallest batches have few subjects. It runs **between GPA
and PCA** (ComBat needs *corresponded* points, since it corrects feature-by-feature): harmonize
the `(S, M, 3)` GPA-aligned coordinates against the batch label, then re-`gpa` + `build_ssm`. Pass
genuine biological size as a **protected continuous covariate** so real variation along it is
regressed out of the batch model and added back, not erased.

### 6. Group analysis (`groups.py`)

Tests whether a binary per-subject label (e.g. healthy vs. pathological, or any two groups)
tracks shape. Everything operates on the `(S, K)` score matrix plus a boolean group vector, so it
is independent of ShapeWorks and unit-testable on synthetic scores.

- **`per_mode_tests`** — *which* modes differ: per-mode **Mann–Whitney U** + **Cohen's d**, with
  **Benjamini–Hochberg FDR** correction across the tested modes (restrict to the leading `top_k`
  to keep FDR power up).
- **`shape_predicts_cancer`** — omnibus *does shape predict the label at all?*: cross-validated
  **ROC AUC** of an L2 logistic model on the leading mode scores (standardised, stratified k-fold,
  out-of-fold probabilities), with a **label-permutation** p-value (fraction of shuffles whose CV
  AUC ≥ the real one). Robust where a parametric multivariate test would be unstable at `S ≈ K`.
- **`group_mean_difference_shape`** — reconstructs each group's mean shape (overall mean + the
  group's mean score projected back through the modes) and their per-point difference, in the
  model's mm frame, to show *where* the groups diverge.
- **`volume_difference`** — raw size gap (Mann–Whitney on volume), reported separately because with
  `scale=False` a shape difference may partly reflect size.

### 7. Export (`export.py`)

**`export_model`** serialises the model — mean, modes, eigenvalues, variance ratios, optional mesh
faces, and JSON provenance metadata — to a single `.npz`. **`load_model`** reloads it into an
`ExportedModel` that depends only on numpy (not scikit-learn or this package), so a separate
program can reconstruct shapes from a handful of weights. `instance(b, pose)` evaluates `x(b)` and
optionally applies a `(scale, rotation, translation)` pose; `clamp` enforces the `±3 sd`
plausibility box.

Because a `scale=True` model is **pose-free and unitless**, matching it to a real anatomy in mm
world coordinates means solving a **7-DOF similarity** (uniform scale + rotation + translation)
alongside the shape weights. **`ExportedModel.fit`** does this by alternating two closed-form
solves until convergence: (1) a **pose** solve (Umeyama similarity carrying the current shape onto
the target), and (2) a **shape** solve (project the pulled-back target onto the modes, rescale to
SD units, clamp). The `units` field records whether scale must be recovered at fit time
(`"normalized"`) or is already baked in (`"mm"`).

### SSM quality metrics (`evaluation.py`)

The three standard Davies et al. (2002) curves, each vs. the number of retained modes **M**:

- **compactness(M)** — cumulative fraction of variance captured (*higher is better*; scale-free).
- **generalization(M)** — **leave-one-out** reconstruction error in mm: PCA on the other `S−1`
  shapes, project the held-out shape onto M modes, measure mean per-particle distance (*lower is
  better*; a rise at high M is overfitting). Matches ShapeWorks Studio's CSV exactly.
- **specificity(M)** — draw random shapes (`b ~ N(0, λ)`) and measure the mean distance to the
  nearest **training** shape (*lower is better*; rises with M as samples wander off the data).

`load_shapeworks_evaluation` reads Studio's own CSVs back into the same container for an overlay.

## Quality control (`qc.py`)

`run_qc(dataset, config) → QCReport` inspects every volume/mask and returns findings tagged
INFO / WARNING / ERROR; **non-destructive and never raises** — it is advisory, and the caller
decides what to exclude. Three check groups:

- **image integrity** — finite values, positive spacing, anisotropy, non-constant image,
  geometry consistency;
- **mask validity** — non-empty, label sanity, connected components / islands, field-of-view
  border contact, physical volume;
- **cross-mask** — one structure inside another, mask over tissue rather than background.

Thresholds (e.g. plausible organ-volume range) live in `QCConfig`; **pass an anatomy-appropriate
config** rather than relying on the defaults. SimpleITK + numpy only.

## Data loading (`io/reader.py`)

`discover_cases(dir)` returns a `Dataset` of `Case` objects, each holding its image sequences,
organ segmentations, and any extra-structure segmentations (one entry per reader annotation).
Input volumes are **MetaImage** (`.mhd`/`.raw`) or **NIfTI** (`.nii`/`.nii.gz`), loaded via
**SimpleITK** (the one reader that handles both). Key behaviours:

- **Filename tokens are authoritative over pixel dtype** when classifying a file as image vs.
  organ vs. extra-structure segmentation — masks are sometimes stored as float but are still
  binary label maps, so never classify by dtype. Tokens are configurable.
- **Layout is auto-detected per directory**: a per-case folder vs. a flat
  `<case-id>.ext` + `<case-id>_segmentation.ext` layout.
- **Flat ids are namespaced by sub-folder**, so the same `CaseNN` under two sub-folders stays
  distinct (recursive discovery otherwise merges geometry-inconsistent cases).
- Geometry mismatches against the reference image are **flagged, not resampled** (resampling is a
  later stage).
s

## Install & commands

src-layout; package at `src/shape_modeling/`, tests in `tests/`. Use a venv (`.venv/` is gitignored).

```bash
pip install -e ".[dev]"          # add .[analysis] for stages 3–7 (scikit-image/-learn, scipy, neuroCombat)
pytest                           # tests (synthetic; ShapeWorks optimiser is an opt-in smoke test)
ruff check src tests && mypy src # lint + types
```

ShapeWorks is **not pip-installable** — install it separately from
<https://sciinstitute.github.io/ShapeWorks/> and run the correspondence step from its environment.
Every other stage runs from the project venv.

## Notebooks (drivers)

Run the correspondence cell from the ShapeWorks environment; every other cell works from the
project venv. The notebooks under `notebooks/` are exploratory drivers that wire the stages
together and carry the confound diagnostics (site / acquisition / annotation), the harmonization
before/after evidence, the group-difference test, and the mean/mode-shape visualisations.

## Scripts (production path)

`scripts/` holds the non-notebook, end-to-end drivers — the production counterpart to the
notebooks, keeping only what the final model needs and writing every result number to an output
folder.

```bash
# 1. Build the final model (from the ShapeWorks conda env). Writes model.npz + CSVs + summary.json.
python scripts/build_prostate_ssm.py \
    --saml SAML --prostate158 Prostate158 --picai PICAI \
    --num-particles 1024 --out ssm_output

# 2. Validate it on a dataset the model never saw (no ShapeWorks needed).
python scripts/validate_ssm.py --model ssm_output/model.npz \
    --data NCI-ISBI --out validation_output --n-modes 15
```

- **`build_prostate_ssm.py`** — discover → cohort → groom + ShapeWorks correspondence (cached) →
  GPA → optional ComBat on acquisition centre → PCA → quality curves → `export_model`. The
  prostate-specific cohort logic (dropping SAML's truncated `I2CVB` site, relabelling SAML by
  centre, joining PI-CAI's csPCa labels) lives here — **copy and edit this script for another
  organ** rather than reusing it as-is.
- **`validate_ssm.py`** — held-out validation: fit the exported model to each unseen mask's surface
  (`fit_surface`, no correspondence needed) and report symmetric surface distance (mm).
- **`fetch_picai.py`** — stdlib-only downloader for PI-CAI AI whole-gland masks + a `labels.csv`
  sidecar. PI-CAI masks are AI-derived, so pooling them is a sensitivity check, not a finding.
