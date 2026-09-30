# Grooming-Referenz: echter Medoid (Thyroid-SSM)

## Warum

`_groom_cohort` richtet jede Form starr auf eine Referenzform aus. Die Referenz kommt aus
`sw.find_reference_image_index`, also aus `MeshUtils::findReferenceMesh` in ShapeWorks.
Bei mehr als 30 Formen vergleicht diese Funktion nur eine Zufallsteilmenge von 30. Das
Minimum wird danach aber über **alle** Formen gesucht. Die nie verglichenen Formen behalten
den Mittelwert 0 und gewinnen deshalb immer. Betroffen sind ShapeWorks ≤ 6.7.0 und `master`
(Stand 2026-09). Beim Thyroid-Modell mit 1054 Lappen ist die Referenz daher sehr
wahrscheinlich einfach Index 0 (`s0002_L`) und kein Medoid. Issue-Text, Repro und Patch
liegen unter `shapeworks_issue/`.

Dieser Job berechnet den echten Medoid nach ShapeWorks' eigener Definition, aber über alle
554 931 Paare. Pro Paar läuft ein starres vtk-ICP von i auf j (10 Iterationen,
Schwerpunktstart), danach wird die mittlere Punkt-zu-Zelle-Distanz berechnet. Pro Form wird
`Summe / (n − 1)` gemittelt, das argmin ist der Medoid. Die Meshes sind dieselben, die
`find_reference_image_index` erzeugt: Grooming-Schritt 1 (`_pregroom_binary`, gemeinsam
genutzt mit `_groom_cohort`) → `computeDT(0.5)` → `toMesh(0.5)`.

Der Job ändert **nichts** am bestehenden Modell. Erst wenn der Medoid bekannt ist, lässt
sich über `ShapeWorksConfig(reference_index=…)` in einer neuen `--work-dir` neu bauen.

## Ablauf (SLURM, `template.sh` → `run_job.py`)

| Task | Array-ID | Was | Ressourcen |
|---|---|---|---|
| `reference_prepare` | 0 | Kohorte wie `build_thyroid_ssm.py` (Split 0.2/Seed 0, 17 Ausschlüsse) neu aufbauen, Fingerprint gegen `cohort.csv` prüfen, 1054 Meshes schreiben | 2 CPU, 8 GB, ≤ 4 h |
| `reference_selfcheck` | 1 | Eigene Paardistanz gegen `findReferenceMesh(meshes[:40], 0)`. Muss übereinstimmen, sonst bricht die Kette ab. Zeigt nebenbei den fehlerhaften Default-Aufruf auf echten Daten | 8 CPU, ≤ 2 h |
| `reference_pairs` | 100–299 | 200 Chunks à ~2775 Paare, maximal 50 gleichzeitig | 1 CPU, 4 GB, ≤ 4 h |
| `reference_reduce` | 2 | Chunks zusammenführen (vollständige Abdeckung wird erzwungen) → Medoid | 1 CPU |

Jeder Schritt startet nur, wenn der vorherige erfolgreich war (`afterok`).

## Auf den Cluster bringen und starten

```bash
# Mac: Stand committen + pushen (CLAUDE.md ist gitignored, bei Bedarf separat kopieren)
git add -A && git commit -m "..." && git push

# Cluster (Login-Node)
cd /home/zore25/Repositories && git clone https://github.com/Zoe-Reinke/ShapeModeling.git
cd ShapeModeling
# Env-Probe von Hand (dasselbe prüft template.sh):
for e in shapemodelrecon shaperecon; do PYTHONPATH=src conda run -n $e python -c \
  "import shapeworks as sw, SimpleITK, shape_modeling.analysis; print('$e ok', getattr(sw,'__version__','?'))"; done
ls /data/Data2/3D_THYROID_US/ShapeModel/data/Thyroid/Totalsegmentator_dataset_v201 | head -3
DRY=1 bash submit_reference_thyroid.sh     # zeigt nur die sbatch-Aufrufe
bash submit_reference_thyroid.sh           # startet die Kette
```

## Ergebnisse (`reference_thyroid_512p_excl/`)

- `selfcheck.json`: `agree` muss `true` sein. `shapeworks_default` ist das Ergebnis des
  fehlerhaften Aufrufs auf den ersten 40 Lappen und bestätigt den Bug empirisch.
- `reference.json`: Medoid (Index, case_id, mittlere Distanz), Rang von Index 0, Top 10.
- `ranking.csv`: alle 1054 Lappen, sortiert nach mittlerer Distanz.
