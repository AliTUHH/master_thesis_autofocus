# docs/notizen/ – Arbeitsnotizen zur Masterarbeit

Alle Notizen liegen als Markdown im Repo, damit sie versioniert, durchsuchbar und beim Zusammenschreiben direkt verwertbar sind. Kein separates Notiz-Tool nötig; wer eine Oberfläche möchte, öffnet den Ordner `docs/` als Vault in [Obsidian](https://obsidian.md) (kostenlos, lokal, reines Markdown – Einstellungen `.obsidian/` dann in `.gitignore` aufnehmen).

## Ordnerstruktur

| Datei / Ordner | Inhalt | Rhythmus |
| --- | --- | --- |
| `experimente.md` | **Experiment-Logbuch** – eine Tabelle, jede Zeile ein Run (Run-Name, Datum, Commit, Daten-Hash, Config-Änderung, Metriken, Beobachtung, nächster Schritt) | nach jedem Lauf |
| `fragen_an_betreuer.md` | Laufende Liste offener Fragen; abgehakte wandern mit Antwort nach unten | fortlaufend |
| `wochen/JJJJ-WW.md` | Wochennotiz (Vorlage `vorlage_wochennotiz.md`) | freitags, 15 min |
| `meetings/JJJJ-MM-TT_betreuer.md` | Protokoll Betreuer-Meeting (Vorlage `vorlage_betreuer_meeting.md`) | pro Meeting, am selben Tag |
| `paper/<citekey>.md` | Paper-Notiz (Vorlage `vorlage_paper_notiz.md`); Dateiname = Citekey aus `thesis/references.bib` | pro gelesenem Paper |

Ordner `wochen/`, `meetings/`, `paper/` beim ersten Gebrauch anlegen.

## Regeln

1. **Citekeys statt Titel.** Paper werden überall mit dem Citekey aus `thesis/references.bib` referenziert, in eckigen Klammern: `[dora2025autofocus]`. So lässt sich jede Notiz später 1:1 in `\cite{dora2025autofocus}` übersetzen.
2. **Jede Zahl hat eine Quelle.** Metriken immer mit Run-Name und Commit-SHA; Literaturwerte mit Citekey und Seitenzahl.
3. **Entscheidungen festhalten, nicht nur Ergebnisse.** Warum wurde ein Ansatz verworfen? Das ist später der Diskussionsteil.
4. **Kurz und datiert.** Lieber fünf Stichpunkte pro Woche als ein Aufsatz pro Monat. Datumsformat `JJJJ-MM-TT`.
5. **Markdown-Konventionen:** Überschriften `##`, Tabellen für Vergleiche, Aufgaben als `- [ ]`, Formeln inline `$Fr = \Delta x^2 / (\lambda (z_{02}-z_{01}) M)$` (gleiche Notation wie in `thesis/main.tex`).
6. Notizen werden mit normalen Commits versioniert (`docs: Wochennotiz 2026-41`), kein PR nötig.

## Von der Notiz zur Thesis

- Paper-Notizen → Kapitel 2/3 (Grundlagen, Stand der Technik) und `docs/02_literatur.md`
- Logbuch + Experiment-Issues → Kapitel 5 (Ergebnisse) und Anhang (Reproduktion)
- Meeting-Protokolle (Entscheidungen) → Kapitel 4 (Methodenwahl begründen) und 6 (Diskussion)
- Wochennotizen → Zeitplan-Reflexion, Vollständigkeits-Check der Forschungsfragen
