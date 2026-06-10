# ShapeModeling

Statistical Shape Model (SSM) pipeline for organ anatomy from MRI volumes and segmentation
masks. Built around the prostate (healthy + cancerous cases), but the readers, QC, and
analysis are organ-agnostic. The headline question: **do glands harbouring cancer have a
systematically different whole-gland shape than benign ones?** — assessed statistically, with
the final model exported for downstream use (segmentation, simulation, augmentation).

> Implementation notes, gotchas, and the full module reference live in **[CLAUDE.md](CLAUDE.md)**.

## Pipeline

1. **Load** MRI + segmentation masks — `src/shape_modeling/io/reader.py` ✅
2. **Quality control** of volumes + masks — `src/shape_modeling/qc.py` ✅
3. **Surface extraction** from masks — `src/shape_modeling/analysis/surface.py` ✅ *(exploratory)*
4. **Correspondence** (ShapeWorks particles) — `src/shape_modeling/analysis/correspondence.py` ✅ *(exploratory)*
5. **SSM construction** (Procrustes + PCA) — `src/shape_modeling/analysis/ssm.py` ✅ *(exploratory)*
6. **Group analysis** (healthy vs. cancer) — `src/shape_modeling/analysis/groups.py` ✅ *(exploratory)*
7. **Model export** — pending

Stages 3–6 are an **exploratory first pass**: good enough to inspect the cohort and answer
"is there a signal?", not the final validated model. The correspondence is throwaway-quality,
so any group effect is *suggestive*, not a biomarker.

## Notebooks (drivers)

Run from the **ShapeWorks** environment (`import shapeworks` + `shapeworks` CLI on PATH) for
the correspondence cell; every other cell works from the project venv.

- `notebooks/explore_shapes.ipynb` — healthy whole-gland SSM, multi-center (Prostate158 +
  SAML). Inspects the shape space and tests the site/acquisition/annotation confounds
  (Kruskal–Wallis, coil mechanism, ComBat-style harmonization).
- `notebooks/explore_shapes_alldata.ipynb` — pooled **anatomical** basis over *all* glands
  (healthy + cancer, multi-center). A descriptive basis, **not** a substrate for the cancer
  question (cancer is confounded with source there).
- `notebooks/explore_cancer_shape.ipynb` — the stage-6 headline test, **within Prostate158**
  (83 cancer + 56 healthy, one protocol) so a group difference can't be a cross-dataset
  confound.
- `notebooks/test_reader.ipynb` — manual reader check.

## Current findings (exploratory)

- **Healthy multi-center basis** has stacked confounds in the leading modes: acquisition site
  (incl. a confirmed receive-coil effect) and annotation protocol (Prostate158 TZ∪PZ union vs
  SAML single contour). Do **not** read raw modes as biology when pooling across sources;
  harmonize on centre and protect gland size as a covariate.
- **Cancer vs. healthy (Prostate158, whole gland, `scale=False`)**: gland shape *modestly*
  separates the groups — omnibus CV AUC ≈ 0.66 (permutation p = 0.004), modes 1–2 differ after
  FDR. Cancer glands are also **smaller** (median 47 vs 59 cm³, p = 0.001), so part of the
  effect is size, not pure shape. Treat as suggestive.

## Data

Input volumes are **MetaImage** (`.mhd`/`.raw`) or **NIfTI** (`.nii`/`.nii.gz`), loaded via
**SimpleITK**. Per-patient directories (`Prostate158/`) and flat layouts (multi-center `SAML/`,
namespaced by sub-folder) are auto-detected. See [CLAUDE.md](CLAUDE.md#data) for the layout
and token rules.

## Install & commands

src-layout; package at `src/shape_modeling/`, tests in `tests/`. Use a venv (`.venv/` is gitignored).

```bash
pip install -e ".[dev]"          # add .[analysis] for stages 3–6 (scikit-image/-learn, scipy)
pytest                           # tests (synthetic; ShapeWorks optimiser is an opt-in smoke test)
ruff check src tests && mypy src # lint + types
```

ShapeWorks is **not pip-installable** — install it separately from
<https://sciinstitute.github.io/ShapeWorks/> and run the correspondence step from its environment.
