<!-- Titel-Konvention: <typ>: <kurze Beschreibung>   (typ = feat | fix | exp | data | docs | thesis | ci | refactor) -->

## Was & Warum
<!-- Was ändert dieser PR, und welches Problem/Issue löst er? -->
Closes #

## Art der Änderung
- [ ] Code (`src/`, `tests/`, `configs/`)
- [ ] Daten-/Simulations-Pipeline (`src/data/`, `configs/data_*.yaml`)
- [ ] Experiment-Ergebnis (nur `results.json`/Figuren, keine Checkpoints/Daten)
- [ ] Dokumentation / Notizen (`docs/`)
- [ ] Thesis-Text (`thesis/`)
- [ ] CI / Tooling (`.github/`, `tools/`, `notebooks/`)

## Tests
- [ ] `python -m pytest -q` lokal grün
- [ ] CI-Workflow grün
- [ ] Smoke-Test ausgeführt (Befehl + Dauer):

## Auswirkung auf Ergebnisse
<!-- Ändert sich das Verhalten von Datenerzeugung, Training oder Evaluation? Dann: -->
- Betroffene Metriken (vorher → nachher, Run-Namen):
- Bestehende Logbuch-Einträge bleiben reproduzierbar? (Config-/Daten-Format kompatibel?)
- Falls nein: neuer Daten-/Config-Versionsstand dokumentiert in `docs/notizen/experimente.md`

## Thesis-Kapitel betroffen
<!-- z. B. 4.2 Datenerzeugung, 5.3 Ablationen; "keins" ist eine gültige Antwort -->
- 

## Checkliste
- [ ] Keine großen Binärdateien (Checkpoints `.pt`, HDF5, Rohdaten) im Diff
- [ ] Neue Abhängigkeiten in `requirements.txt`/`environment.yml` eingetragen
- [ ] Doku angepasst, falls sich ein CLI-Befehl oder eine Konvention geändert hat
