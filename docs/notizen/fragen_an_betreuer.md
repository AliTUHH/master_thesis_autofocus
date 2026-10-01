# Fragen an die Betreuer

Laufende Liste. Neue Fragen oben anfügen, mit Datum und Priorität (**A** = blockiert Arbeit, **B** = wichtig für Planung, **C** = Nice-to-know). Beantwortete Fragen mit Antwort, Datum und Quelle (Meeting-Protokoll) in den Abschnitt „Beantwortet“ verschieben.

## Offen

### Daten & Physik
- [ ] **A** · 2026-10-01 · **Zugang zu Messdaten (P05):** Gibt es bereits Hologramm-Datensätze (Flat-Field-korrigiert) mit bekannter Geometrie, die wir für Tests nutzen dürfen? Format (HDF5/TIFF), Umfang, Speicherort (DESY-Dateisystem/Maxwell, Download möglich?), Datenschutz/Publikationsregeln.
- [ ] **A** · 2026-10-01 · **Referenz-Fr für Messdaten:** Woher kommt die „Wahrheit“ für gemessene Hologramme – modellbasierter Autofokus (Dora et al. 2025), Motorpositionen, oder beides? Wie groß ist die Unsicherheit dieser Referenz (sonst können wir die Netzgenauigkeit nicht sinnvoll bewerten)?
- [ ] **B** · 2026-10-01 · **Typische Geometrie an P05:** Bereiche für z01, z02, Energie, Pixelgröße, Detektor (Bildgröße), damit `configs/data_p05.yaml` realistisch ist. Soll das Netz nur z01 bei festem z02 schätzen oder Fr allgemein (mehrere Geometrien)?
- [ ] **B** · 2026-10-01 · **Phantome/Proben für die Simulation:** Welche Objektklassen sind repräsentativ (biologische Proben, Metallschäume, Testmuster)? Gibt es in HoloForge vorbereitete Phantome oder Probenbibliotheken, die wir nutzen sollen?
- [ ] **B** · 2026-10-01 · **Rausch- und Beleuchtungsmodell:** Welche Störungen müssen in der Simulation vorkommen, damit der Transfer auf Messdaten realistisch ist (Photonenstatistik, Detektor-PSF, Flat-Field-Reste, Probe-Wellenfront)?
- [ ] **C** · 2026-10-01 · **Zielgröße:** Präferenz für z01 [m], Fr oder log Fr als Regressionsziel? Gibt es Erfahrungen, welche Parametrisierung für die Optimierung günstiger ist?

### Rechenressourcen
- [ ] **A** · 2026-10-01 · **DESY Maxwell-Cluster:** Kann der Student einen Account erhalten (über J. Dora)? Welche GPU-Partition, Speicherkontingent, Zugang per SSH/VPN, Jupyter-Hub? Ist holowizard dort als Modul vorhanden?
- [ ] **B** · 2026-10-01 · **TUHH-Ressourcen:** Gibt es am Institut/der TUHH GPU-Rechner oder ein HPC-Angebot für Studierende? Antragsweg?
- [ ] **C** · 2026-10-01 · Ist Colab/Kaggle für mittlere Experimente aus Sicht der Betreuer in Ordnung (keine Messdaten dorthin hochladen?) – Vertraulichkeit der Daten klären.

### Thesis & Formalia
- [ ] **A** · 2026-10-01 · **Sprache der Arbeit:** Deutsch oder Englisch? (Vorlage kann beides; Entscheidung früh treffen, Literatur ist englisch.)
- [ ] **B** · 2026-10-01 · **Vorlage:** Hat das Institut eine eigene LaTeX-Vorlage oder Vorgaben (Titelblatt, Logo, Erklärungstext)? Sonst nutzen wir `thesis/` (eigenes Skelett, an TUHH-Vorgaben angelehnt) – ist das akzeptabel?
- [ ] **B** · 2026-10-01 · **Umfang und Gliederung:** Erwarteter Seitenumfang? Ist die Gliederung in `thesis/main.tex` (Einleitung, Grundlagen, Stand der Technik, Methoden, Experimente, Diskussion, Fazit) passend? Gewichtung Physik vs. ML?
- [ ] **B** · 2026-10-01 · **Abgabeformalitäten:** Anmeldedatum/Bearbeitungszeit, Abgabeform (PDF-Upload, gedruckte Exemplare?), Eidesstattliche Erklärung (offizieller Wortlaut), Regelung zu KI-Werkzeugen, Kolloquium/Vortrag (Dauer, Termin).
- [ ] **C** · 2026-10-01 · **Veröffentlichung:** Darf der Code öffentlich auf GitHub bleiben (ist es bereits)? Zenodo-DOI für den Code zum Abschluss gewünscht? Ggf. gemeinsames Paper?

### Zusammenarbeit
- [ ] **B** · 2026-10-01 · **Meeting-Rhythmus:** Wöchentlich/zweiwöchentlich? Mit wem (TUHH- und DESY-Seite getrennt oder gemeinsam)? Bevorzugter Kanal für kurze Fragen (E-Mail, Mattermost/Slack, GitHub-Issues)?
- [ ] **C** · 2026-10-01 · Sollen die Betreuer Lesezugriff auf GitHub-Issues/Projects-Board bekommen, um den Fortschritt zu verfolgen?
- [ ] **C** · 2026-10-01 · Gibt es HoloWizard-Entwickler-Dokumentation oder Beispiele für die Simulations-API (HoloForge), die über die öffentliche Doku hinausgehen?

## Beantwortet

| Datum | Frage (Kurzform) | Antwort | Quelle |
| --- | --- | --- | --- |
| | | | `meetings/JJJJ-MM-TT_betreuer.md` |
