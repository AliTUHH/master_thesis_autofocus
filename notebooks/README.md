# notebooks/

Notebooks sind hier **nur Einstiegspunkte für Rechenumgebungen** (Colab/Kaggle) und für explorative Analysen. Die eigentliche Logik lebt in `src/` und wird über die CLI aufgerufen (`python -m src.data.generate_data`, `python -m src.train`, `python -m src.evaluate`) – so bleiben Experimente reproduzierbar und testbar.

| Notebook | Zweck | Öffnen |
| --- | --- | --- |
| `colab_training.ipynb` | Komplette Pipeline in Google Colab: Repo klonen → Python-3.11-venv mit `uv` (Colab läuft mit 3.12, `holowizard` braucht 3.11) → Daten erzeugen (`--out data/processed/small`) → Training mit `--run-name` (→ `runs/<run>/results.json`, `best.pt`, Plots, `tb/`) → TensorBoard inline → Evaluation (→ `runs/<run>/eval_test/eval_results.json`) → Ergebnisse nach Google Drive | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AliTUHH/master_thesis_autofocus/blob/main/notebooks/colab_training.ipynb) |

Direktlink: <https://colab.research.google.com/github/AliTUHH/master_thesis_autofocus/blob/main/notebooks/colab_training.ipynb>

## Regeln

1. **Ohne Ausgaben committen** (*Bearbeiten → Alle Ausgaben löschen* bzw. `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`), damit Diffs lesbar bleiben und keine großen Bilder im Git landen.
2. Keine Ergebnisse, die in die Thesis sollen, aus Notebook-Zellen heraus „per Hand“ erzeugen – immer über die CLI, damit Run-Name, Config und Commit-SHA nachvollziehbar sind (siehe `docs/04_werkzeuge_und_workflow.md`, Abschnitt 4).
3. Neue Notebooks nach Zweck benennen (`explore_<thema>.ipynb`, `colab_<zweck>.ipynb`) und oben in eine Markdown-Zelle schreiben, was sie tun und welchen Datensatz/Run sie voraussetzen.
4. Für Kaggle dasselbe Notebook verwenden: *File → Import Notebook → GitHub* und die Zelle zum Drive-Mount überspringen (Kaggle hat Drive nicht; Ergebnisse stattdessen als Kaggle-Output speichern oder herunterladen).

## Validierung

Das Notebook wurde mit `nbformat` erzeugt und validiert (`nbformat.validate`, Format 4.5). Zum erneuten Prüfen:

```bash
python -c "import nbformat; nb = nbformat.read('notebooks/colab_training.ipynb', as_version=4); nbformat.validate(nb); print('ok')"
```
