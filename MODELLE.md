# Modelldokumentation — finale SSMs (Prostata & Thyroid)

Stand: 2026-09-15. Zweck: vollständiger, nachvollziehbarer Record, **welche Daten** und
**welche Parameter** in die beiden finalen Statistical Shape Models eingegangen sind.
Alle Zahlen stammen aus den `summary.json` / `*.csv` der jeweiligen Output-Ordner, nicht
aus Erinnerung.

## 0. Welches Modell ist jeweils final?

| | **Prostata** | **Thyroid (Einzel-Lappen)** | **Thyroid mm-Variante** |
|---|---|---|---|
| Modelldatei | `ssm_output_scale/model.npz` | `thyroid_ssm_output_excl/model.npz` | `thyroid_ssm_output_excl_mm/model.npz` |
| Gebaut mit | `scripts/build_prostate_ssm.py` | `scripts/build_thyroid_ssm.py` | dito, `--no-scale` |
| Gebaut am | 2026-06-16 | 2026-08 | 2026-09-15 |
| Trainingsformen | 336 Drüsen (336 Patient:innen) | 1054 Lappen (527 Subjekte) | **dieselben** 1054 Lappen |
| Partikel | 1024 | 512 | 512 (**identische** Partikel) |
| GPA | `scale=True` → `units="normalized"` | `scale=True` → `units="normalized"` | **`scale=False` → `units="mm"`** |
| Harmonisierung | **keine** (`--no-harmonize`, bewusst) | entfällt (ein Datensatz) | entfällt |
| Held-out-Validierung | 59 Fälle, **1.66 mm** mean | 279 Lappen, **1.01 mm** median | **noch nicht gelaufen** (s. 3.4) |
| Superseded | – | `thyroid_ssm_output/` (544 Subj., ohne QC-Ausschlüsse) | – |

Alle Modell-Outputs sind **gitignored** (nicht im Repo-History). Die Rohdaten liegen
seit dem Cleanup 2026-08 off-repo:

- Prostata: `/Volumes/Data2/3DTRUSPro/PublicMRI/` (`PICAI/`, `Prostate158/`, `SAML/`, `testset/`)
- Thyroid: `/Volumes/Data2/3D_THYROID_US/ShapeModel/data/Thyroid/Totalsegmentator_dataset_v201/`

---

# 1. Prostata-SSM

## 1.1 Artefakte

| Pfad | Inhalt |
|---|---|
| `ssm_output_scale/model.npz` | exportiertes Modell (mean, modes, eigenvalues, faces, Metadaten) |
| `ssm_output_scale/cohort.csv` | **Fallliste**: case_id, source, centre, volume_cm3, is_cancer (336 Zeilen) |
| `ssm_output_scale/eigenvalues.csv` | pro Mode: eigenvalue, Varianzanteil, kumulativ |
| `ssm_output_scale/evaluation.csv` | compactness / generalization_mm / specificity_mm pro Modenzahl |
| `ssm_output_scale/summary.json` | alle Kennzahlen des Laufs |
| `ssm_output_scale/particles.npz`, `shape_model.swproj` | Korrespondenz + ShapeWorks-Studio-Projekt |
| `shapeworks_work_final_1024p/` | Grooming-Output + Partikeldateien |
| `validation_testset/` | Held-out-Validierung (summary.json + validation.csv) |

## 1.2 Trainingsdaten (336 Drüsen: 131 Krebs / 205 gesund)

| Quelle | n | davon Krebs | Masken-Provenienz | Dateien |
|---|---|---|---|---|
| Prostate158 | 139 | 83 | human, T2-Reader 1, TZ∪PZ-**Union** | `Prostate158/<id>/t2_anatomy_reader1.nii.gz` |
| PI-CAI | 100 | 48 | **KI-generiert** (Whole-Gland) | `PICAI/<id>_segmentation.nii.gz` + `PICAI/labels.csv` |
| SAML:BMC | 30 | 0 | human, ein Whole-Gland-Kontur | `SAML/BMC/CaseNN_Segmentation.nii.gz` |
| SAML:RUNMC | 30 | 0 | human | `SAML/RUNMC/…` |
| SAML:UCL | 13 | 0 | human | `SAML/UCL/…` |
| SAML:BIDMC | 12 | 0 | human | `SAML/BIDMC/…` |
| SAML:HK | 12 | 0 | human | `SAML/HK/…` |

Geometrie = **ganze Drüse**, `organ_segs[0] > 0` (erster Reader, binarisiert).
Volumen im Kohortenumfang: 14.8–312.9 cm³, Median 46.3 cm³
(`RUNMC/Case16` = 312.9 cm³ ist eine echte BPH-Drüse, kein Segmentierungsfehler).
Krebslabel kommt nur aus dem Vorhandensein einer Läsionsmaske bzw. aus `labels.csv` —
verglichen wird immer die Gesamtdrüse. Vollständige Fall-IDs: **Anhang A**.

**Ausschlüsse:** die komplette SAML-Site **`I2CVB`** (`EXCLUDE_SITES` in
[build_prostate_ssm.py](scripts/build_prostate_ssm.py)) — ihre Whole-Gland-Masken sind planar
abgeschnitten (Annotations-Crop), wodurch die ICP-Registrierung divergiert (Welt-COM ~60 mm
versetzt). PROMISE12 wurde früher zugunsten von SAML ersetzt und ist nicht Teil des finalen Modells.

## 1.3 Pipeline & Parameter

1. **Discovery**: `discover_cases()` pro Datensatz getrennt (eindeutige Source-Labels),
   flache IDs werden nach Unterordner namespaced.
2. **Kohorte**: `build_grouped_cohort()` (behält gesund + Krebs), I2CVB gedroppt,
   SAML nach Akquisitionszentrum relabelt, PI-CAI-Labels aus `labels.csv` gejoint.
3. **Grooming** (ShapeWorks, ganze Kohorte gemeinsam; `ShapeWorksConfig`-Defaults):
   `antialias(30 it) → resample(Linear, iso 1.0 mm) → binarize → COM-Align →
   Medoid-Referenz + rigides ICP (200 it, iso-Level 0.0, auf das Referenzgitter resampelt,
   pad 20 vx) → fitRegion(bbox + 20 vx) → computeDT(0.5) → gaussianBlur(σ 1.3 mm)`.
4. **Korrespondenz**: ShapeWorks `optimize`, **1024 Partikel**, **1000 Iterationen**,
   relative_weighting 1.0, Regularisierung 1000 → 10, narrow_band 30.
   Ergebnis: **spread ratio 0.985** (Gate: ≤1.75).
5. **GPA**: `gpa(scale=True)` — Translation, Rotation **und Skalierung** entfernt
   → Modell ist unitlos; Größe wird erst beim Fit über eine 7-DOF-Similarity rekonstruiert.
6. **ComBat**: **übersprungen** (`--no-harmonize`). Begründung: die Endorektalspulen-Deformation
   (SAML BMC/BIDMC/HK) soll erhalten bleiben, weil sie der TRUS-Sonden-Deformation ähnelt und
   das Modell als Ultraschall-Prior dienen soll. Trade-off siehe 1.6.
7. **PCA**: `build_ssm` → 335 Moden über 1024 Punkte.
8. **Evaluation**: `evaluate_ssm` auf den World-Partikeln (mm), seed 0.
9. **Export**: `export_model` mit Convex-Hull-Faces (nur zur Ansicht) + Provenienz-Metadaten.
   **Achtung**: `build_prostate_ssm.py` schreibt seit 2026-09-16 stattdessen
   `reconstruct_surface(model.mean).faces` (echte Rekonstruktion, siehe Kap. 3.7). Das
   *ausgelieferte* `ssm_output_scale/model.npz` stammt vom Lauf davor und enthält weiterhin die
   Convex Hull — der Reproduktionsbefehl unten erzeugt also ein besseres `faces`-Array als das
   archivierte Modell.

**Reproduktionsbefehl** (aus der ShapeWorks-Conda-Env, `PATH=/Applications/ShapeWorks/bin:$PATH`):

```bash
python -u scripts/build_prostate_ssm.py \
  --saml SAML --prostate158 Prostate158 --picai PICAI \
  --num-particles 1024 --iterations 1000 \
  --no-harmonize \
  --work-dir shapeworks_work_final_1024p \
  --out ssm_output_scale
```

## 1.4 Modellkennzahlen

- 335 Moden / 1024 Punkte; Mode 1 = 25.1 %, Mode 2 = 16.3 %, Mode 3 = 12.2 % Varianz
- 90 / 95 / 99 % Varianz bei **7 / 15 / 60** Moden
- Generalisierung (LOO): 1.25 mm @ 7 Moden, **0.89 mm @ 15 Moden**, 0.29 mm bei allen Moden
- Spezifität: 1.72 mm @ 7, 1.93 mm @ 15 Moden
- Empfohlene Arbeitsdimension: **~15 Moden**

## 1.5 Held-out-Validierung (`validation_testset/`)

Daten, die das Modell nie gesehen hat (`/Volumes/Data2/3DTRUSPro/PublicMRI/testset/`):
**NCI-ISBI Prostate-3T** (`Prostate3T-01-XXXX_segmentation.nrrd`, 30 Fälle) und
**MSD Decathlon Task05** (`prostate_NN_segmentation.nii.gz`, 32 Fälle).

Ablauf: `clean_mask → extract_surface → ExportedModel.fit_surface` (ICP + Modengewichts-Solve,
eigene Pseudo-Korrespondenz), **15 Moden**, max_sd 3.0, 4 ICP-Restarts, Volumen-Gate 10–200 cm³.

- 62 gefunden, **3 per Volumen-Gate ausgeschlossen** (Prostate3T_01_0017 313.0 cm³,
  prostate_16 315.4 cm³, prostate_38 202.3 cm³) → **59 gefittet**
- **mean 1.66 mm**, Median 1.69, p95 2.01, max 2.10 mm (hd95 ~3.2 mm)
- Prostate3T 1.70 mm vs. MSD 1.63 mm — konsistent über beide externen Sets
- ≈1.9× der In-sample-Generalisierung (0.89 mm) = normale Optimismus-Lücke, kein Overfitting
- liegt auf dem Inter-Observer-Niveau der Prostata (~1–2 mm ASSD); Testmasken sind 3.6 mm
  schichtdick, d. h. die Metrik hat einen Diskretisierungsboden

```bash
python -u scripts/validate_ssm.py --model ssm_output_scale/model.npz \
  --data testset --n-modes 15 --out validation_testset
```

## 1.6 Einschränkungen (bewusst in Kauf genommen)

- **Keine saubere anatomische Basis**: ohne ComBat mischen die führenden Moden 1–3
  Site / Spule / Annotationsprotokoll mit Anatomie. Wird ein reines Anatomiemodell gebraucht,
  denselben Lauf **ohne** `--no-harmonize` wiederholen (ComBat auf Zentrum, Größe als Kovariate).
- **Gemischte Maskenprovenienz**: 100 von 336 Masken sind KI-generiert (PI-CAI). Dieser Lauf ist
  damit streng genommen die PI-CAI-inklusive Variante = Sensitivitätscheck, nicht die rein
  human-annotierte Referenz.
- **Annotationsprotokolle unterschiedlich**: Prostate158 = TZ∪PZ-Union, SAML/PI-CAI = eine
  Whole-Gland-Kontur (persistenter Confound in Moden 2–3 der explorativen Läufe).
- Modell ist unitlos; absolute Größe nur aus dem gefitteten Pose-Scale bzw. aus `fit.points`.
- `model.faces` ist in diesem Artefakt eine Convex Hull — nur zur Ansicht, kein faithful Mesh
  (dafür `warp_surface` oder ein Rebuild, der jetzt `reconstruct_surface` nutzt, siehe Kap. 3.7).

---

# 2. Thyroid-Lappen-SSM

## 2.1 Artefakte

| Pfad | Inhalt |
|---|---|
| `thyroid_ssm_output_excl/model.npz` | exportiertes Modell |
| `thyroid_ssm_output_excl/cohort.csv` | **Formenliste**: case_id, side, volume_cm3 (1054 Zeilen) |
| `thyroid_ssm_output_excl/heldout.txt` | 141 Held-out-Subjekte |
| `thyroid_ssm_output_excl/excluded.txt` | 17 per Sichtprüfung ausgeschlossene Subjekte |
| `thyroid_ssm_output_excl/eigenvalues.csv`, `evaluation.csv`, `summary.json` | Kennzahlen |
| `thyroid_ssm_output_excl/shapeworks_model/` | ShapeWorks-Studio-Projekt zum Ansehen |
| `shapeworks_work_thyroid_final_512p_excl/` | Grooming + Partikel |
| `thyroid_validation_knownside_output_excl/` | Held-out-Validierung |
| `thyroid_ssm_briefing_for_3dus.txt` | Anwendungs-Briefing für die 3D-US-Rekonstruktion |

## 2.2 Trainingsdaten (1054 Lappen: 527 links + 527 rechts, 527 Subjekte)

- Quelle: **TotalSegmentator v201** (CT), `Thyroid/Totalsegmentator_dataset_v201/<sID>/segmentations/`
- Dateien pro Subjekt: `thyroid_left.nii.gz` + `thyroid_right.nii.gz`, erzeugt aus TotalSegmentators
  `thyroid_gland.nii.gz` durch Split an der Trachea-Mittellinie
  ([thyroid_modeling/split_thyroid.py](thyroid_modeling/split_thyroid.py))
- Kein Isthmus, **ein Lappen pro Form**; Lappenvolumen 0.50–41.1 cm³, Median 6.46 cm³
- Datensatz enthält 705 Subjektordner; daraus Kandidaten = Subjekte mit **beiden** Lappen
  ≥ `MIN_VOLUME_CM3 = 0.3` → Split in 141 Held-out + Training, dann 17 QC-Ausschlüsse
- Vollständige Listen: **Anhang B**

**Ausschlüsse:**
- **17 Subjekte** nach visueller Kontrolle (s0299, s0303, s0556, s0592, s0808, s0809, s0824,
  s0844, s0898, s0927, s0950, s0970, s1008, s1147, s1281, s1400, s1420): grob asymmetrische
  Paare (z. B. s0927 18.1/44.7 cm³) und implausibel kleine Lappen, die durch das 0.3-cm³-Gate
  rutschten. Liste + Begründungen: [thyroid_excluded_subjects.txt](thyroid_excluded_subjects.txt).
- **Wichtiges Detail:** der Ausschluss wird **nach** dem Held-out-Split angewendet. Der Split ist
  `rng.permutation(candidates)`; filtert man vorher, verschiebt sich jede Zuordnung und ein bereits
  validiertes `heldout.txt` wird ungültig. So bleiben die 141 Held-out-Subjekte bit-identisch.
- `s0650` (Whole-Gland-Fusion, kontralateraler Lappen leer) liegt im Held-out-Set und ist in der
  Validierung als `data_qc_exclusions` geführt — kein Modellfehler.

## 2.3 Pipeline & Parameter

1. **Laden**: direkt per SimpleITK in handgebaute `CohortEntry`-Objekte (TotalSegmentator-Dateinamen
   passen nicht zu den Reader-Tokens des Prostata-Pfads).
2. **Spiegelung**: jeder **rechte** Lappen wird an `MIRROR_AXIS = 2` (Array-Achse x = L/R) gespiegelt,
   damit beide Seiten dieselbe Chiralität haben (Procrustes kann nicht spiegeln) → Modell ist
   **seitenagnostisch** im linken Chiralitätsrahmen.
3. **ROI-Crop + Rezentrierung**: jeder Lappen wird auf seine Bounding Box + `ROI_MARGIN = 8` Voxel
   zugeschnitten und der COM auf den Weltursprung gelegt (ohne das divergiert das COM/ICP-Grooming
   im vollen Hals-/Thorax-CT-Gitter).
4. **Grooming**: wie Prostata, aber `iso_spacing = 1.5 mm` (= effektiver Voxel-Boden des Modells).
5. **Korrespondenz**: ShapeWorks, **512 Partikel**, **1000 Iterationen** → **spread ratio 0.882**.
6. **GPA**: `gpa(scale=True)` → unitlos, Größe normalisiert.
7. **PCA**: 1053 Moden über 512 Punkte. **Evaluation** + **Export** wie beim Prostatamodell.

**Reproduktionsbefehl** (ShapeWorks-Conda-Env; `shapeworks`-CLI liegt in
`/Applications/ShapeWorks/bin`, **nicht** in der Env → PATH voranstellen):

```bash
PATH=/Applications/ShapeWorks/bin:$PATH python -u scripts/build_thyroid_ssm.py \
  --data Thyroid/Totalsegmentator_dataset_v201 \
  --num-particles 512 --iterations 1000 \
  --holdout-frac 0.2 --holdout-seed 0 \
  --exclude-file thyroid_excluded_subjects.txt \
  --work-dir shapeworks_work_thyroid_final_512p_excl \
  --out thyroid_ssm_output_excl
```

## 2.4 Modellkennzahlen

- 1053 Moden / 512 Punkte; Mode 1 = 38.9 %, Mode 2 = 18.7 %, Mode 3 = 7.6 % Varianz
- 90 / 95 / 99 % Varianz bei **5 / 10 / 90** Moden
- Generalisierung (LOO): 0.86 mm @ 5, 0.59 mm @ 10, **0.47 mm @ 15** Moden, 0.12 mm bei allen
- Spezifität: 0.99 mm @ 5, 1.21 mm @ 15 Moden
- Empfohlene Arbeitsdimension: **10–15** Moden (weniger = stärkerer Prior für verrauschte US-Daten)

## 2.5 Held-out-Validierung (`thyroid_validation_knownside_output_excl/`)

141 Held-out-Subjekte, Variante **known-side** (Seite ist bekannt → deterministisch spiegeln,
nicht raten), 15 Moden, max_sd 3.0, 4 Restarts, Volumen-Gate 0.3–40 cm³.

- 282 Lappen gefunden, 2 leere Masken per Gate raus → **280 gefittet**; ohne den bestätigten
  Fusionsfall `s0650_R`: **279 Lappen**
- **Median 1.011 mm, mean 1.031, p95 1.382, max 3.010 mm**, Konvergenz **93.6 %**
- pro Seite: L Median 0.968 mm (n=139), R Median 1.028 mm (n=141)
- Gegenüber dem Vormodell (544 Subjekte, ohne die 17 Ausschlüsse; Median 1.111 / mean 1.148 /
  p95 1.550) besser auf **274 von 279 Lappen**, Wilcoxon p = 3.2e-47 → der QC-Ausschluss ist ein
  echter Deployment-Gewinn, nicht kosmetisch
- 1.0 mm liegt am 1.5-mm-Voxelboden der CT-Masken — nahe am Machbaren für diese Eingangsauflösung
- schlechtester verbleibender Fall: `s1133_L` (3.0 mm) — Kandidat für die nächste QC-Runde

```bash
python -u scripts/validate_thyroid_ssm_knownside.py \
  --model thyroid_ssm_output_excl/model.npz \
  --data Thyroid/Totalsegmentator_dataset_v201 \
  --holdout-list thyroid_ssm_output_excl/heldout.txt \
  --n-modes 15 --out thyroid_validation_knownside_output_excl
```

## 2.6 Einschränkungen

- **CT-trainiert**, nicht Ultraschall: rein geometrischer Prior, kein Bildmodell (Domain Gap).
- **Nur ein Lappen**, kein Isthmus, keine ganze Drüse. Links und rechts getrennt fitten,
  rechte Lappen vor dem Fit an x spiegeln und danach zurückspiegeln.
- Größe ist normalisiert (`scale=True`); absolute Volumina nur aus dem gefitteten mm-Surface.
- Kropf-Tail (> ~20 cm³) am Rand der Trainingsverteilung → dort seltener Konvergenz.
- Knoten/Zysten sind nicht repräsentiert und erscheinen als Fit-Residuum.

## 2.7 Zugehörige Analyse (nicht Teil des Modells)

Links-/Rechts-Lappen-Asymmetrie, [explore_thyroid_lobes.ipynb](notebooks/explore_thyroid_lobes.ipynb):
rechter Lappen größer (+0.861 cm³ Median, p=1.8e-26); größenfreie, gepaarte FDR-Signifikanz in allen
Top-10-Moden (min p=1.4e-21) bei kleiner absoluter L/R-Formdifferenz (~0.70 mm). Spiegel-Artefakt
ausgeschlossen (globale Affin-Fraktion 0.11). Mode-*Nummern* verschieben sich zwischen Läufen —
nur Vorhandensein + Lokalisation sind interpretierbar.

---

# 3. Thyroid-Lappen-SSM — mm-Variante (`scale=False`)

Zweite Ausleitung desselben Modells für das Projekt, das SSM-Fit und 3D-US-Rekonstruktion
**gemeinsam** optimiert. Dort ist Größe Teil des Priors und darf nicht vom Pose-Scale
wegabsorbiert werden.

## 3.1 Verhältnis zum Modell aus Kapitel 2

Kohorte, Ausschlüsse, Held-out-Split, Grooming und Korrespondenz sind nicht nur "gleich
parametrisiert", sondern **dieselben Objekte**: der Lauf hat den Partikel-Cache
`shapeworks_work_thyroid_final_512p_excl/particles.npz` wiederverwendet. Verifiziert: ein
`scale=True`-Rebuild aus diesem Cache reproduziert `thyroid_ssm_output_excl/model.npz`
bit-identisch (`mean` maxdiff 0.0, Eigenwerte identisch). Einziger Unterschied ist
`gpa(scale=False)` zwischen Korrespondenz und PCA.

```bash
PATH=/Applications/ShapeWorks/bin:$PATH python -u scripts/build_thyroid_ssm.py \
  --data Thyroid/Totalsegmentator_dataset_v201 \
  --num-particles 512 --iterations 1000 \
  --holdout-frac 0.2 --holdout-seed 0 \
  --exclude-file thyroid_excluded_subjects.txt \
  --no-scale \
  --work-dir shapeworks_work_thyroid_final_512p_excl \
  --out thyroid_ssm_output_excl_mm
```

(Der Lauf vom 2026-09-15 lief ohne gemountete Rohdaten direkt aus dem Cache; obiger Befehl ist
der äquivalente Vollpfad und trifft dieselbe Cache-Zeile.)

## 3.2 Modellkennzahlen

| | scale=True (Kap. 2) | **scale=False (mm)** |
|---|---|---|
| `units` | `normalized` | `mm` |
| Mean-Bounding-Box | 0.083 × 0.075 × 0.116 (unitlos) | **31.5 × 22.7 × 19.9 mm** |
| Mode 1 / 2 / 3 | 38.9 / 18.7 / 7.6 % | **63.9 / 14.9 / 6.9 %** |
| 90 / 95 / 99 % Varianz | 15 / 37 / 205 Moden | **5 / 10 / 79 Moden** |

Mode 1 ist hier ein reiner **Größenmode**: Walk von −3 auf +3 sd geht über Meshvolumen
0.18 → 4.59 → 19.98 cm³ (längste Achse 8.9 → 31.5 → 56.8 mm; Volumina aus dem neuen
`faces`-Mesh, Kap. 3.7 — die früher dokumentierten 0.19 → 4.80 → 22.51 cm³ waren Hüllvolumina). Die Kohorte belegt auf Mode 1 nur
etwa −2.5 … +2.2 sd (p1…p99); 6 von 1054 Lappen liegen jenseits |3| sd. **Eine Mode geht für
Größe drauf** — 15 Moden hier entsprechen grob 14 Moden im normalisierten Modell.

`evaluation.csv` ist **identisch** zum Modell aus Kapitel 2: `evaluate_ssm` läuft per Design auf
den World-Partikeln in mm und ist vom GPA-Flag unabhängig (Generalisierung 0.59 mm @ 10,
0.465 mm @ 15 Moden).

## 3.3 Rigider Fit (Library-Änderung, nötig für diese Variante)

`units="mm"` war bis 2026-09-15 reine Dokumentation: `fit`/`fit_surface` haben **immer** eine
7-DOF-Similarity gelöst, hätten die mm-Größe beim Fit also wieder wegskaliert. Jetzt lesen beide
`ExportedModel.units` und pinnen bei `"mm"` `scale = 1` (auch in den ICP-Restart-Posen);
`fit_scale=True/False` übersteuert. Gegenprobe auf 4 Trainingslappen (zufällig rigide verschoben,
15 Moden): rigider Fit `pose-scale` exakt 1.0 bei 0.33–1.14 mm — auf 3 von 4 Lappen *besser* als
der 7-DOF-Fit des normalisierten Modells trotz eines Freiheitsgrads weniger, weil der Größenprior
informativ ist.

## 3.4 Validierung — offen

Die Held-out-Zahl aus 2.5 (**1.011 mm Median**) gilt **nicht** für dieses Modell: sie wurde mit
7-DOF-Similarity gemessen, der rigide Fit hat einen Freiheitsgrad weniger. Die 141 Held-out-
Subjekte sind unverändert (`thyroid_ssm_output_excl_mm/heldout.txt`), die Messung ist also bereit
— sie braucht nur die gemountete Rohdatenplatte:

```bash
python -u scripts/validate_thyroid_ssm_knownside.py \
  --model thyroid_ssm_output_excl_mm/model.npz \
  --data Thyroid/Totalsegmentator_dataset_v201 \
  --holdout-list thyroid_ssm_output_excl_mm/heldout.txt \
  --n-modes 15 --out thyroid_validation_knownside_output_excl_mm
```

## 3.5 Auslieferung: `thyroid_ssm_mm_export/`

Self-contained Bundle (13.1 MB) von `scripts/make_ssm_bundle.py`, nur numpy + scipy nötig:
`model.npz`, `thyroid_ssm.py` (generierter Standalone-Loader), `BRIEFING.txt`
(= `thyroid_ssm_briefing_mm_for_3dus.txt`), `cohort.csv`, `heldout.txt`, `excluded.txt`,
`eigenvalues.csv`, `evaluation.csv`, `summary.json`. Der Loader wird per `ast` aus
`export.py`/`surface.py` gehoben (nicht handkopiert) und beim Bundling dagegen verifiziert —
bit-identische Gewichte, Pose und Distanz.

Rauchtest aus dem Bundle heraus: realer Lappen, rigide verschoben, 0.6 mm Rauschen, 25 % der
Oberfläche entfernt (Schallschatten) → Fit 1.00 mm gegen die sichtbaren Punkte, **1.32 mm gegen
die vollständige, unverdeckte Wahrheit** — der Prior füllt den Schatten.

## 3.6 Befund aus dem Export: ein kontaminierter Trainingslappen

`s0044_L` hat alle 512 Partikel auf einen **0.37 × 0.30 × 0.52 mm** Klumpen kollabiert, obwohl
seine Maske 0.739 cm³ misst — ein degeneriertes Optimizer-Ergebnis. Der Spread-Ratio-Gate greift
nicht, weil er der **Median** über Subjekte ist (0.882); ein einzelner Ausreißer verschwindet
darin. Der Lappen steckt im Training **beider** Thyroid-Varianten.

Gemessener Einfluss ist klein: ohne ihn verschiebt sich der Mittelwert um 0.016 mm und die
führenden Varianzanteile um ~0.3 Prozentpunkte (mm) bzw. 38.9 → 40.8 % (normalisiert). Er sitzt
aber bei **−4.1 sd auf Mode 1** und ist der Grund, warum der Mode-1-Tiefpunkt auf unmögliche
0.19 cm³ extrapoliert. Empfehlung im Briefing: `b1 >= -2.5` als Zusatzschranke statt der
symmetrischen ±3-sd-Box. Kandidat für die nächste QC-Runde (mit `s1133_L`, 3.0 mm) — ein
**Per-Subjekt**-Spread-Gate statt des Medians würde ihn fangen.

## 3.7 `faces`: echte Rekonstruktion statt Convex Hull (2026-09-16)

**Problem**: `model.faces` war `ConvexHull(model.mean).simplices` — byte-identisch nachgewiesen.
Eine Hülle kann die konkave Medialfläche des Lappens nicht darstellen, und nur **393 der 512**
Korrespondenzpunkte lagen überhaupt auf ihr (119 Punkte in keinem einzigen Dreieck). Die
Docstring von `export_model` hatte `hull_surface(model.mean).faces` als Beispiel geführt; beide
Build-Skripte hatten das übernommen.

**Fix**: neue Funktion `reconstruct_surface(points)` in `analysis/surface.py` — Delaunay-
*Sculpting*: aus der Delaunay-Tetraedrisierung der Punkte wird jeweils das Randtetraeder mit dem
größten Umkugelradius abgetragen, solange sein **Umkugelmittelpunkt außerhalb** des Körpers
liegt (ein Tetraeder, das eine Konkavität überbrückt, hat ihn außen; eines im Organinneren nahe
der Medialachse, also innen). Das braucht **keinen** Alpha-Parameter und passt sich der
Organdicke an — eine reine Radiusschwelle kann das nicht: bei reiner Oberflächenabtastung hat
das Innere Tetraeder mit Umkugelradius ≈ halbe Organdicke, ein Alpha klein genug für eine echte
Konkavität höhlt das Organ also aus (Hantel aus zwei Kugeln: −46 % Volumen; Thyroidlappen bei
`alpha = 1.5 ×` Punktabstand: −55 %). Abgetragen wird nur, wenn der Rand eine geschlossene
2-Mannigfaltigkeit bleibt und kein Punkt ins Innere zurückfällt.

**Re-Export**: `build_thyroid_ssm.py` neu gelaufen (2026-09-16) aus dem Partikel-Cache heraus —
ShapeWorks wird dafür nicht gebraucht, der Lauf lief im Repo-`.venv`. Kohorte, Held-out-Split,
Ausschlüsse und Metadaten sind identisch; `mean`, `components` und Eigenwerte reproduzieren das
Modell vom 2026-09-15 bis auf ≤1.5e-12 (BLAS-Rauschen zwischen den Umgebungen). Material geändert
hat sich **nur `faces`** — plus die *stochastische* Specificity-Spalte (<0.01 mm, Metrik-Rauschen;
Compactness und Generalization sind bis 1e-15 identisch). Das alte Modell liegt als
`model_convexhull_faces_2026-09-15.npz.bak` neben dem neuen auf der NAS.

**Ergebnis** (`thyroid_ssm_output_excl_mm/model.npz`, unverändert außer `faces`):

| | Convex Hull (alt) | **reconstruct_surface (neu)** |
|---|---|---|
| Dreiecke | 782 | **1020** (= 2·512 − 4, Genus 0) |
| genutzte Punkte | 393 / 512 | **512 / 512** |
| mittlerer symm. Fehler vs. groomed DT | 0.40 mm (max 3.05) | **0.25 mm (max 0.68)** |
| Volumen vs. groomed DT | +1.3 % (lokal falsch) | **−1.7 %** |
| watertight / manifold / orientiert | ja / ja / **nein** | **ja / ja / ja** |

Verifiziert gegen die groomed Distanztransformationen, auf denen die Partikel optimiert wurden
(29 Subjekte), und gegen eine Rekonstruktion *pro Subjekt* — die feste Mittelwert-Topologie ist
so gut wie eine subjektspezifische (0.25 mm beide).

**Wichtige Einschränkung — Volumen ist ein *groomed* Volumen**: das Mesh gibt die Modellform
korrekt wieder, aber die groomed Formen sind systematisch kleiner als die CT-Masken
(Antialias + 1.5-mm-Resample + DT-Blur erodieren). Median Mesh/CT-Masken-Verhältnis **0.71**
(p5 0.40, p95 0.83); das Mittel-Mesh hat 4.59 cm³ gegenüber 6.46 cm³ Median-Maskenvolumen der
Kohorte. Das ist **keine** Mesh-Eigenschaft (Mesh/groomed-DT = 0.98), sondern die Skala des
Modells selbst.

**Zweite Einschränkung — eine feste Topologie faltet auf Instanzen**: auf dem Mittelwert ist das
Mesh per Konstruktion selbstdurchdringungsfrei, aber dieselbe Facettenliste für andere Formen
wiederzuverwenden erzeugt Selbstdurchdringungen — sauber bei 89 % der 5-Moden-, 77 % der
10-Moden-, 65 % der 60-Moden-Stichproben (Fehlerfall typisch 1–7 Dreieckspaare von 1020; reale
Trainingslappen schlechter). Ursache ist die **Korrespondenz**, nicht das Mesh: eine
Rekonstruktion der *eigenen* Partikel eines Lappens ist immer sauber, und die alte Hülle faltet
auf denselben Formen stärker. `qc_mesh(surface) -> MeshQC` (aus `mesh.py` des
ShapeModelReconstruction-Projekts portiert, in numpy/scipy statt Open3D) prüft Mannigfaltigkeit,
Wasserdichtheit, Selbstdurchdringung und Orientierung; `build_thyroid_ssm.py` gatet damit den
Export (Mittelwert hart, 50 Stichproben als Warnung).

---

# 4. Umgebung & Nutzung

- **Build** erfordert ShapeWorks: Python-Modul in `~/miniconda3/envs/shapeworks`, die
  **CLI v6.7.0** aber unter `/Applications/ShapeWorks/bin/shapeworks` → in nicht-interaktiven
  Shells `PATH=/Applications/ShapeWorks/bin:$PATH` voranstellen, sonst bricht der Lauf **nach**
  dem Grooming mit `FileNotFoundError: 'shapeworks'` ab. Python mit `-u` starten, wenn in ein Log
  umgeleitet wird. Grooming wird **nicht** gecacht (nur `particles.npz`) — ein Neustart groomt neu.
- **Notebooks**: eigener Jupyter-Kernel für die Conda-Env verwenden, nicht das Projekt-`.venv`.
- **Nutzung der Modelle braucht kein ShapeWorks und kein sklearn** — nur numpy:

```python
from shape_modeling.analysis import load_model   # oder der generierte Standalone-Loader
m = load_model("thyroid_ssm_output_excl/model.npz")
fit = m.fit_surface(cloud_mm, n_modes=15)        # nicht-korrespondierte (N,3) mm-Punktwolke
recon, code, pose = fit.points, fit.weights, fit.pose
```

- `x = mean + Σ b_k · √λ_k · φ_k`, `b_k` in SD-Einheiten, plausibel ist `|b_k| ≤ 3` (`m.clamp`).
- **Die Pose-Freiheitsgrade folgen `m.units`**: `"normalized"` (Prostata, Thyroid Kap. 2) → 7-DOF-
  Similarity, Scale wird mitgelöst; `"mm"` (Thyroid Kap. 3) → **6-DOF rigid**, `scale = 1` gepinnt,
  damit der Größenprior nicht wegskaliert wird. `fit_scale=` übersteuert.
- Für eine Weitergabe an ein fremdes Projekt `scripts/make_ssm_bundle.py` nutzen statt
  `export.py` von Hand zu kopieren — `export.py` importiert aus dem Package und ist **nicht**
  copy-paste-fähig; der generierte Loader ist es.

---

## Anhang A — Prostata: verwendete Fälle (336)

Maschinenlesbar: `ssm_output_scale/cohort.csv` (case_id, source, centre, volume_cm3, is_cancer).

### Prostate158 — Krebs (83)
Ordner `Prostate158/<id>/`, Maske `t2_anatomy_reader1.nii.gz`

```
020, 021, 025, 027, 029, 030, 031, 032, 033, 035, 036, 038, 039, 041, 043, 045, 046, 048, 050,
051, 052, 053, 057, 058, 059, 060, 063, 064, 065, 066, 073, 074, 075, 076, 077, 078, 079, 080,
081, 082, 086, 088, 089, 090, 092, 096, 099, 100, 101, 103, 104, 105, 106, 107, 109, 110, 111,
112, 113, 114, 118, 121, 122, 124, 125, 130, 131, 132, 133, 136, 137, 138, 140, 143, 146, 148,
150, 152, 153, 154, 156, 157, 158
```

### Prostate158 — gesund (56)

```
022, 023, 024, 026, 028, 034, 037, 040, 042, 044, 047, 049, 054, 055, 056, 061, 062, 067, 068,
069, 070, 071, 072, 083, 084, 085, 087, 091, 093, 094, 095, 097, 098, 102, 108, 115, 116, 117,
119, 120, 123, 126, 127, 128, 129, 134, 135, 139, 141, 142, 144, 145, 147, 149, 151, 155
```

### SAML:BIDMC (12) — alle gesund
Dateien `SAML/BIDMC/<CaseNN>_Segmentation.nii.gz`

```
BIDMC/Case00, BIDMC/Case02, BIDMC/Case03, BIDMC/Case04, BIDMC/Case05, BIDMC/Case06,
BIDMC/Case07, BIDMC/Case08, BIDMC/Case09, BIDMC/Case10, BIDMC/Case11, BIDMC/Case12
```

### SAML:BMC (30) — alle gesund
Dateien `SAML/BMC/<CaseNN>_Segmentation.nii.gz`

```
BMC/Case00, BMC/Case01, BMC/Case02, BMC/Case03, BMC/Case04, BMC/Case05, BMC/Case06, BMC/Case07,
BMC/Case08, BMC/Case09, BMC/Case10, BMC/Case11, BMC/Case12, BMC/Case13, BMC/Case14, BMC/Case15,
BMC/Case16, BMC/Case17, BMC/Case18, BMC/Case19, BMC/Case20, BMC/Case21, BMC/Case22, BMC/Case23,
BMC/Case24, BMC/Case25, BMC/Case26, BMC/Case27, BMC/Case28, BMC/Case29
```

### SAML:HK (12) — alle gesund
Dateien `SAML/HK/<CaseNN>_Segmentation.nii.gz`

```
HK/Case38, HK/Case39, HK/Case40, HK/Case41, HK/Case42, HK/Case43, HK/Case44, HK/Case45,
HK/Case46, HK/Case47, HK/Case48, HK/Case49
```

### SAML:RUNMC (30) — alle gesund
Dateien `SAML/RUNMC/<CaseNN>_Segmentation.nii.gz`

```
RUNMC/Case00, RUNMC/Case01, RUNMC/Case02, RUNMC/Case03, RUNMC/Case04, RUNMC/Case05,
RUNMC/Case06, RUNMC/Case07, RUNMC/Case08, RUNMC/Case09, RUNMC/Case10, RUNMC/Case11,
RUNMC/Case12, RUNMC/Case13, RUNMC/Case14, RUNMC/Case15, RUNMC/Case16, RUNMC/Case17,
RUNMC/Case18, RUNMC/Case19, RUNMC/Case20, RUNMC/Case21, RUNMC/Case22, RUNMC/Case23,
RUNMC/Case24, RUNMC/Case25, RUNMC/Case26, RUNMC/Case27, RUNMC/Case28, RUNMC/Case29
```

### SAML:UCL (13) — alle gesund
Dateien `SAML/UCL/<CaseNN>_Segmentation.nii.gz`

```
UCL/Case01, UCL/Case26, UCL/Case27, UCL/Case28, UCL/Case29, UCL/Case30, UCL/Case31, UCL/Case32,
UCL/Case33, UCL/Case34, UCL/Case35, UCL/Case36, UCL/Case37
```

### PI-CAI — csPCa-positiv (48)
Dateien `PICAI/<id>_segmentation.nii.gz` (KI-Masken), Label aus `PICAI/labels.csv`

```
10005_1000005, 10008_1000008, 10012_1000012, 10013_1000013, 10019_1000019, 10021_1000021,
10029_1000029, 10032_1000032, 10036_1000036, 10040_1000040, 10043_1000043, 10044_1000044,
10048_1000048, 10050_1000050, 10053_1000053, 10059_1000059, 10074_1000074, 10078_1000078,
10079_1000079, 10085_1000085, 10094_1000094, 10097_1000097, 10100_1000100, 10103_1000103,
10104_1000104, 10106_1000106, 10107_1000107, 10110_1000110, 10112_1000112, 10117_1000117,
10118_1000118, 10121_1000121, 10132_1000134, 10134_1000136, 10135_1000137, 10140_1000142,
10148_1000150, 10157_1000160, 10164_1000167, 10168_1000171, 10170_1000173, 10171_1000174,
10182_1000185, 10202_1000206, 10209_1000213, 10211_1000215, 10212_1000216, 10214_1000218
```

### PI-CAI — csPCa-negativ (52)

```
10000_1000000, 10001_1000001, 10002_1000002, 10003_1000003, 10004_1000004, 10006_1000006,
10007_1000007, 10009_1000009, 10010_1000010, 10011_1000011, 10014_1000014, 10015_1000015,
10016_1000016, 10017_1000017, 10018_1000018, 10020_1000020, 10022_1000022, 10023_1000023,
10024_1000024, 10025_1000025, 10026_1000026, 10027_1000027, 10028_1000028, 10030_1000030,
10031_1000031, 10033_1000033, 10034_1000034, 10035_1000035, 10037_1000037, 10038_1000038,
10039_1000039, 10041_1000041, 10042_1000042, 10045_1000045, 10046_1000046, 10047_1000047,
10049_1000049, 10051_1000051, 10052_1000052, 10054_1000054, 10055_1000055, 10056_1000056,
10057_1000057, 10058_1000058, 10060_1000060, 10061_1000061, 10062_1000062, 10063_1000063,
10064_1000064, 10065_1000065, 10066_1000066, 10067_1000067
```


## Anhang B — Thyroid: Subjekte

Maschinenlesbar: `thyroid_ssm_output_excl/cohort.csv` (1054 Lappen), `heldout.txt`, `excluded.txt`.

### Trainings-Subjekte (527, je 1 linker + 1 rechter Lappen = 1054 Formen)
Dateien `Thyroid/Totalsegmentator_dataset_v201/<id>/segmentations/thyroid_left.nii.gz` bzw. `thyroid_right.nii.gz`

```
s0002, s0003, s0011, s0019, s0024, s0025, s0028, s0029, s0032, s0035, s0040, s0044, s0046,
s0049, s0056, s0059, s0061, s0065, s0067, s0071, s0079, s0080, s0081, s0086, s0091, s0095,
s0100, s0110, s0111, s0112, s0114, s0123, s0124, s0130, s0138, s0140, s0141, s0145, s0146,
s0149, s0152, s0154, s0162, s0163, s0169, s0171, s0172, s0175, s0181, s0185, s0189, s0191,
s0196, s0217, s0223, s0224, s0226, s0230, s0231, s0234, s0239, s0240, s0241, s0253, s0259,
s0263, s0277, s0278, s0279, s0281, s0285, s0288, s0289, s0292, s0310, s0315, s0327, s0332,
s0334, s0344, s0345, s0349, s0354, s0358, s0359, s0362, s0363, s0365, s0371, s0372, s0373,
s0374, s0375, s0377, s0378, s0379, s0380, s0382, s0383, s0392, s0393, s0394, s0396, s0398,
s0402, s0407, s0408, s0412, s0414, s0420, s0421, s0422, s0423, s0424, s0425, s0429, s0430,
s0431, s0433, s0436, s0447, s0452, s0453, s0455, s0456, s0457, s0459, s0462, s0463, s0466,
s0467, s0469, s0470, s0472, s0476, s0477, s0478, s0481, s0482, s0483, s0484, s0485, s0487,
s0488, s0490, s0491, s0493, s0494, s0498, s0499, s0501, s0502, s0503, s0507, s0508, s0510,
s0515, s0516, s0519, s0521, s0526, s0531, s0535, s0536, s0537, s0543, s0550, s0552, s0557,
s0559, s0561, s0565, s0568, s0571, s0572, s0574, s0576, s0578, s0580, s0583, s0585, s0587,
s0593, s0597, s0602, s0604, s0610, s0612, s0613, s0616, s0617, s0618, s0621, s0622, s0624,
s0625, s0626, s0627, s0628, s0632, s0635, s0636, s0637, s0638, s0639, s0641, s0642, s0643,
s0644, s0645, s0648, s0649, s0654, s0656, s0657, s0658, s0659, s0661, s0662, s0663, s0669,
s0674, s0675, s0680, s0681, s0686, s0688, s0690, s0691, s0692, s0695, s0697, s0698, s0700,
s0702, s0703, s0704, s0708, s0709, s0711, s0714, s0716, s0718, s0720, s0723, s0726, s0727,
s0729, s0731, s0733, s0734, s0735, s0736, s0737, s0739, s0740, s0743, s0750, s0751, s0754,
s0756, s0759, s0763, s0764, s0765, s0769, s0770, s0771, s0773, s0778, s0780, s0784, s0785,
s0788, s0789, s0795, s0796, s0797, s0798, s0801, s0802, s0804, s0807, s0810, s0811, s0812,
s0814, s0816, s0820, s0821, s0822, s0829, s0830, s0834, s0835, s0836, s0842, s0845, s0847,
s0849, s0851, s0860, s0863, s0865, s0866, s0867, s0868, s0869, s0870, s0871, s0875, s0878,
s0880, s0884, s0889, s0890, s0894, s0896, s0897, s0899, s0903, s0904, s0908, s0909, s0910,
s0912, s0913, s0914, s0916, s0918, s0919, s0923, s0924, s0925, s0928, s0930, s0931, s0933,
s0940, s0941, s0945, s0946, s0953, s0954, s0955, s0957, s0959, s0960, s0961, s0973, s0974,
s0976, s0978, s0982, s0983, s0986, s0992, s0994, s1001, s1004, s1006, s1007, s1010, s1011,
s1012, s1013, s1015, s1016, s1017, s1018, s1031, s1032, s1033, s1034, s1035, s1036, s1039,
s1040, s1042, s1043, s1044, s1045, s1052, s1058, s1059, s1060, s1061, s1062, s1063, s1066,
s1067, s1069, s1070, s1071, s1073, s1075, s1077, s1083, s1085, s1088, s1089, s1090, s1093,
s1094, s1101, s1102, s1105, s1107, s1110, s1111, s1112, s1113, s1115, s1119, s1120, s1123,
s1124, s1129, s1132, s1135, s1140, s1142, s1143, s1145, s1146, s1152, s1155, s1157, s1159,
s1161, s1162, s1163, s1164, s1165, s1167, s1168, s1171, s1172, s1173, s1174, s1178, s1182,
s1184, s1185, s1189, s1190, s1192, s1195, s1197, s1201, s1206, s1207, s1208, s1209, s1210,
s1220, s1221, s1223, s1225, s1228, s1231, s1233, s1240, s1242, s1245, s1248, s1261, s1262,
s1264, s1267, s1268, s1270, s1272, s1273, s1274, s1275, s1276, s1277, s1278, s1283, s1287,
s1289, s1292, s1295, s1296, s1297, s1298, s1299, s1304, s1305, s1308, s1310, s1311, s1314,
s1316, s1319, s1323, s1326, s1327, s1330, s1332, s1334, s1335, s1336, s1338, s1340, s1343,
s1344, s1348, s1353, s1354, s1361, s1363, s1364, s1365, s1369, s1370, s1371, s1372, s1377,
s1379, s1382, s1387, s1388, s1395, s1397, s1404
```

### Held-out-Subjekte (141) — NICHT im Training, nur für die Validierung

```
s0004, s0021, s0036, s0037, s0043, s0050, s0063, s0070, s0076, s0102, s0115, s0136, s0167,
s0170, s0184, s0197, s0208, s0254, s0271, s0282, s0295, s0312, s0322, s0326, s0331, s0333,
s0346, s0350, s0364, s0367, s0369, s0370, s0400, s0405, s0411, s0413, s0437, s0438, s0439,
s0440, s0444, s0446, s0461, s0497, s0504, s0511, s0517, s0518, s0522, s0527, s0530, s0546,
s0549, s0551, s0553, s0589, s0591, s0598, s0605, s0606, s0619, s0623, s0629, s0650, s0660,
s0664, s0667, s0672, s0687, s0696, s0717, s0719, s0747, s0752, s0753, s0775, s0777, s0790,
s0794, s0833, s0843, s0853, s0857, s0874, s0885, s0892, s0915, s0934, s0939, s0949, s0952,
s0963, s0975, s0977, s0985, s0991, s0999, s1003, s1022, s1023, s1028, s1029, s1038, s1046,
s1051, s1079, s1082, s1086, s1099, s1100, s1114, s1127, s1133, s1138, s1144, s1154, s1169,
s1176, s1199, s1218, s1222, s1224, s1226, s1230, s1235, s1238, s1247, s1249, s1255, s1269,
s1285, s1318, s1321, s1329, s1337, s1349, s1350, s1352, s1362, s1380, s1384
```

### Per Sichtprüfung ausgeschlossen (17) — Begründungen in `thyroid_excluded_subjects.txt`

```
s0299, s0303, s0556, s0592, s0808, s0809, s0824, s0844, s0898, s0927, s0950, s0970, s1008,
s1147, s1281, s1400, s1420
```
