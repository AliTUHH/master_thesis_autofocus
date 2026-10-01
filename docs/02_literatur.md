# Literaturübersicht – Lernbasierter Autofokus für Röntgen-Nahfeldholographie

Stand: Oktober 2026. Nummerierung [1]–[6] wie in der Ausschreibung; weitere Quellen fortlaufend. Jede Quelle trägt einen **Verifikationsgrad**:
- **Volltext** – der vollständige Text (bzw. eine nahezu vollständige Textfassung) wurde in dieser Recherche gelesen; Zahlen stammen direkt daraus.
- **Abstract** – nur Abstract/erste Seite gelesen; Inhaltsangaben beschränken sich auf das, was dort steht oder in gelesenen Volltexten über die Quelle gesagt wird.
- **Metadaten** – nur bibliographische Angaben (Titel, Autoren, Journal, DOI) geprüft, meist über die Referenzlisten gelesener Volltexte; Inhaltsangaben sind bewusst knapp und als Vorwissen gekennzeichnet.

Nichts in dieser Liste ist erfunden; Quellen, die sich nicht belegen ließen, stehen am Ende unter „Verworfene Kandidaten“.

---

## A. Pflichtquellen aus der Ausschreibung

### [1] Dora et al. 2024 – ASRM

**Zitat:** J. Dora, M. Möddel, S. Flenner, C. G. Schroer, T. Knopp, J. Hagemann, „Artifact-suppressing reconstruction of strongly interacting objects in X-ray near-field holography without a spatial support constraint“, *Optics Express* 32(7), 10801–10828 (2024). DOI: [10.1364/OE.514641](https://doi.org/10.1364/OE.514641). Code/Daten: Zenodo 10.5281/zenodo.8349365 (laut Referenzliste des Papers).
**Verifikation:** Volltext (Textfassung nahezu vollständig; Abbildungen nicht gesehen).
**Was ist es:** Methodenpaper zum Phase Retrieval aus einem einzigen Hologramm ohne räumlichen Support; Grundlage des HoloWizard-Core-Rekonstruktors.
**Kernidee:** Erweiterung des PGD-Algorithmus „refAP“ (direkte Rekonstruktion des projizierten Brechungsindex $\tilde O$, dadurch keine $2\pi$-Phasenmehrdeutigkeit) um Vorverarbeitung (Spiegel-Padding, Fensterung, Intensitätsoffset $A_0$ in der Beleuchtung $\psi_\mathrm{exit} = \exp(i\tilde O)A_0 e^{i\phi_0}$), getrennte Regularisierung von Phase und Absorption, Stabilisierung hoher Frequenzen und Nesterov-beschleunigten Gradientenschritt, kombiniert mit einem Multigrid-Schema.
**Kernergebnisse:** Deutlich weniger Artefakte und kürzere Rechenzeit als unmodifiziertes refAP; Demonstration auf mehreren P05-Datensätzen mit stark wechselwirkenden Objekten (Phasenschübe weit jenseits $2\pi$).
**Relevanz:** ASRM ist (a) der innere Löser im Autofokus [2], (b) der Rekonstruktor, mit dem du die Downstream-Qualität deiner $z_{01}$-Schätzung bewertest, (c) Quelle der Nebenbedingungsmenge $\Omega_P$, die dem MFE-Kriterium zugrunde liegt.
**Übernehmen:** Notation ($\tilde O = \varphi + i\mu$, $\Omega_P$, $A_0$), Datenterm in der Amplitude ($\sqrt I$), Multigrid-Parameter als Standard für Referenzrekonstruktionen.

### [2] Dora et al. 2025 – Modellbasierter Autofokus (Stand der Technik)

**Zitat:** J. Dora, M. Möddel, S. Flenner, J. Reimers, B. Zeller-Plumhoff, C. G. Schroer, T. Knopp, J. Hagemann, „Model-based autofocus for near-field phase retrieval“, *Optics Express* 33(4), 6641–6657 (2025). DOI: [10.1364/OE.544573](https://doi.org/10.1364/OE.544573). Open Access (CC BY). Code: Zenodo 10.5281/zenodo.8349364 (= HoloWizard, siehe [5]).
**Verifikation:** Volltext (DESY-PUBDB-Fassung vollständig gelesen, inkl. Tabellen 1–3 und Appendix).
**Was ist es:** Methodenpaper, das das Autofokusproblem der NFH als geschachteltes Optimierungsproblem formuliert, ein neues Fokuskriterium (MFE) einführt und es gegen sechs Literaturkriterien auf Simulation und fünf P05-Messdatensätzen vergleicht.
**Kernidee:** Eine mit falschem $z_{01}$ rekonstruierte Welle enthält Fresnel-Säume außerhalb der physikalischen Menge $\Omega_P$; deren Projektion macht die Welle inkonsistent zum gemessenen Hologramm. Der *Model Fit Error* $\mathrm{MFE} = \||\mathcal D^{\hat z_{12}}_\mathrm{Fr}(\hat\psi_P)| - \sqrt{I_\mathrm{det}}\|_2^2$ (Gl. 14/15) misst diese Inkonsistenz und ist damit ein Kriterium, das Rekonstruktion, Vorwärtsmodell und Messung gemeinsam nutzt. Optimierung: $z_{01}^\ast = \operatorname{argmin}_{\hat z_{01}\in[z^\mathrm{est}\pm\Delta_M]} G(\operatorname{argmin}_{\tilde O\in\Omega_P}\mathcal L(\tilde O,\hat z_{01}))$ (Gl. 22) mit Nelder–Mead, Simplex-Initialisierung an den Intervallenden, Projektion auf das Intervall, Abbruch bei Simplexlänge 100 µm; innerer Löser ASRM [1] mit den Stufen 700/300/500 Iterationen bei 16×/4×/4× Downsampling (Tab. 3).
**Kernergebnisse/Zahlen:** $\Delta_M = \pm5$ mm Messunsicherheit; Fehler in $z_{02}$ vernachlässigbar, $\pm5$ mm in $z_{01}$ ändert $\mathrm{Fr}$ signifikant (Fig. 3). Simulation (drei Phantome, 201 Stützstellen, rauschfrei): MFE, GRA, GoG, ToG v-förmig; VAR ohne Peak; LAP und SPEC verrauscht/inkonsistent. Messdaten (Spinnenhaar 11 keV/79.4 mm, Kaktusnadel 17 keV/284.1 mm, Zahn 17 keV/81.0 mm, Mg-Draht 11 keV/470.8 mm, Mg-Draht in Flusszelle 11 keV/329.3 mm; $z_{02}$ 19.65–19.91 m; $\mathrm{Fr}$ $7.8\cdot10^{-5}$ bis $4.7\cdot10^{-4}$): alle Literaturkriterien zeigen Nebenpeaks oder keinen Peak, MFE nur „minor saddle points“. Autofokus-Testlauf (Tab. 1): Konvergenz auf den visuell bestimmten Wert bis $\le0.1$ mm bei allen fünf Objekten, 9–13 Probepunkte, 6 s (Gitter 8192² → 4× down) bzw. 16 s (14 336² → 4× down) pro Rekonstruktion, insgesamt „a few minutes“. Hardware nicht angegeben.
**Grenzen (explizit oder implizit):** Minuten Laufzeit; lokaler Optimierer, Startintervall nötig; Abhängigkeit von ASRM-Hyperparametern/$A_0$/Padding; nur fünf Messobjekte; Simulation ohne Rauschen; nur $z_{01}$ optimiert; keine Unsicherheit; Referenz ist visuelle Fokussierung.
**Relevanz:** Die Baseline, die du in Genauigkeit und Laufzeit schlagen bzw. beschleunigen willst; liefert zugleich die Referenz-$z_{01}$ für Messdaten und die Anforderungen an Fokuskurven.
**Übernehmen:** Problemformulierung (Gl. 16–22), Unsicherheitsmodell $\pm5$ mm als Prior-Breite, Zielgenauigkeit 0.1 mm, Metrik „Zahl der Rekonstruktionen“ für den Hybridansatz, Tab. 2 als Katalog realistischer Setups für die Simulation.

### [3] Paganin et al. 2002 – Einzelabstands-Phase-Retrieval für homogene Objekte

**Zitat:** D. Paganin, S. C. Mayo, T. E. Gureyev, P. R. Miller, S. W. Wilkins, „Simultaneous phase and amplitude extraction from a single defocused image of a homogeneous object“, *Journal of Microscopy* 206(1), 33–40 (2002). DOI: [10.1046/j.1365-2818.2002.01010.x](https://doi.org/10.1046/j.1365-2818.2002.01010.x).
**Verifikation:** Abstract/Metadaten; Formel über Volltexte von Lohse et al. 2020 [19] und Zuo et al. 2020 [25] sowie Referenzen in [1], [2] bestätigt.
**Was ist es:** Das meistzitierte Phase-Retrieval-Verfahren der propagationsbasierten Röntgenbildgebung („Paganin-Methode“).
**Kernidee:** Für ein homogenes Objekt (konstantes $\delta/\beta$) sind Phase und Absorption beide proportional zur projizierten Dicke; die Transport-of-Intensity-Gleichung lässt sich dann für einen Abstand geschlossen lösen: $T = -\frac{1}{\mu_\mathrm{lin}}\ln\mathcal F^{-1}\big[\mathcal F[I/I_0]/(1+\pi\lambda z(\delta/\beta)|f|^2)\big]$ – ein stabiler Tiefpassfilter, ein FFT-Paar.
**Relevanz:** Zeigt, wie unmittelbar $\mathrm{Fr}$ (über $\lambda z$) in jede Rekonstruktion eingeht; sehr schnelle Referenzrekonstruktion für schwache/homogene Phantome; Einstieg in das Thema laut Ausschreibung (Task 1a).
**Übernehmen:** Begriff des TIE-Regimes (kleine Propagation, erste CTF-Nullstelle außerhalb der relevanten Frequenzen), Einsicht, dass ein falsches $\lambda z$ den Filter verschiebt.

### [4] Fienup 1982 – Phase-Retrieval-Algorithmen im Vergleich

**Zitat:** J. R. Fienup, „Phase retrieval algorithms: a comparison“, *Applied Optics* 21(15), 2758–2769 (1982). DOI: [10.1364/AO.21.002758](https://doi.org/10.1364/AO.21.002758).
**Verifikation:** Volltext.
**Was ist es:** Klassiker, der iterative Phase-Retrieval-Verfahren (Gerchberg–Saxton, Error Reduction, Input-Output/HIO) mit Gradientenverfahren vergleicht.
**Kernidee:** Alternierende Erzwingung von Nebenbedingungen im Objektraum (Support, Nichtnegativität) und Messraum (bekannter Fourier-Betrag). Error Reduction konvergiert monoton, ist aber eng mit Steepest Descent verwandt und stagniert; Input-Output-Varianten (HIO) und konjugierte Gradienten konvergieren in der Praxis viel schneller.
**Relevanz:** Konzeptionelle Wurzel von ASRM (projizierter Gradient = Gradientenschritt + Projektion auf Nebenbedingungen) und der Idee, Nebenbedingungsverletzungen als Fehlermaß zu nutzen (MFE).
**Übernehmen:** Sprachgebrauch (Projektionen, Fehlermetrik im Messraum), Verständnis, warum Nebenbedingungen Eindeutigkeit herstellen.

### [5] HoloWizard – Python-Framework für Online-Rekonstruktion

**Zitat:** J. Dora, S. Flenner, A. Lopes Marinho, J. Hagemann, „A Python framework for the online reconstruction of X-ray near-field holography data“, Zenodo (2024). DOI: [10.5281/zenodo.8349364](https://doi.org/10.5281/zenodo.8349364) (so in [2] zitiert; Konzept-DOI, löst laut DataCite auf die aktuelle Version 1.3.1 (2024, Versions-DOI 10.5281/zenodo.14024980) auf; [1] zitiert 10.5281/zenodo.8349365 = Versions-DOI 1.0.0 (2023)). Die installierte Version 3.0.6 ist separat unter dem Namen „HoloWizard“ veröffentlicht (Konzept-DOI 10.5281/zenodo.16275927, Version 3.0.6: 10.5281/zenodo.19560147). Lokal installiert: `holowizard` 3.0.6 unter `/tmp/holo311`.
**Verifikation:** Metadaten über die Referenzlisten von [1], [2] und die Ausschreibung; der **Quellcode** wurde vollständig gelesen (Pakete `core`, `forge`, `livereco`, `pipe`).
**Was ist es:** Das Softwarepaket, das ASRM [1], den Autofokus [2], den Simulator HoloForge und die Online-Infrastruktur (Livereco-Server, Pipelines) bündelt.
**Kerninhalte (aus dem Code):** `core.api.functions.find_focus.find_focus` (öffentliche Autofokus-API), `core.find_focus.find_focus_z01` (scipy Nelder–Mead, `xatol = z01_tol = 0.1` mm, `initial_simplex` = Intervallenden), Varianten mit $a_0$-Suche und Flatfield-Korrektur, `focus_loss_metrics` (VAR/SPEC/GRA/LAP/ToG/GoG), `core.models.cone_beam.ConeBeam.get_fr/get_z01`, `core.models.fresnel_propagator_torch`, Multistage-Rekonstruktion (`reconstruct_multistage`), Beispielskripte (`core/scripts/examples/find_focus/magnesium_wire.py`, `focus_series_singledim`), HoloForge (`forge.utils.calc_Fr`, `forge.experiment.setup.NFHSetup/NFHConstantDistSetup`, `NFHSimulation`, `PhantomGenerator` mit `xraylib`, `ProbeGenerator`, `HDF5Labeller`, `forge/scripts/generate_data.py`, `forge/configs/default.json`).
**Relevanz:** Dein Simulator, deine Baseline, dein Integrationsziel.
**Übernehmen:** Einheitenkonvention (mm/keV), $\mathrm{Fr}$-Formel, HDF5-Layout (`images/hologram`, `metadata/setup/{Fr,z01,z02,energy,detector_px_size,...}`), Parameterobjekte `Measurement(z01, z01_confidence)`, `Options(z01_tol)`, `Padding(a0)`.

### [6] Deistler et al. 2025 – Simulation-Based Inference: A Practical Guide

**Zitat:** M. Deistler, J. Boelts, P. Steinbach, G. Moss, T. Moreau, M. Gloeckler, P. L. C. Rodrigues, J. Linhart, J. K. Lappalainen, B. K. Miller, P. J. Gonçalves, J.-M. Lueckmann, C. Schröder, J. H. Macke, „Simulation-Based Inference: A Practical Guide“, arXiv:2508.12939 (2025). URL: https://arxiv.org/abs/2508.12939.
**Verifikation:** Volltext (arXiv-HTML).
**Was ist es:** Praxisleitfaden für SBI mit neuronalen Netzen: Workflow von Prior und Simulator über Training bis Diagnostik, mit `sbi`-Codebeispielen.
**Kernidee:** Simulator als implizite Likelihood; Lernen von Posterior (NPE), Likelihood (NLE) oder Likelihood-Ratio (NRE) aus simulierten Paaren $(\theta, x)$; amortisierte Inferenz für neue Beobachtungen; Embedding-Netze für hochdimensionale Daten (Bilder) end-to-end mittrainierbar. Ausführlicher Diagnostikteil: Prior Predictive Checks, Posterior Predictive Checks, Simulation-Based Calibration (SBC), Expected Coverage, TARP, lokale Tests (L-C2ST, posterior SBC), Umgang mit Modellfehlspezifikation, NPE-Ensembles gegen Überkonfidenz, Hinweise zu Dichteschätzern (Normalizing Flows, Flow Matching, Diffusion) und zur Wahl der Zahl der Simulationen.
**Relevanz:** Direkt in der Ausschreibung genannt (Task 2c „estimates for the parameter $z_{01}$“); liefert den Rahmen für WP 4 und die Kalibrierungsmetriken in WP 5.
**Übernehmen:** Schrittfolge (Prior → Simulator → Checks → Training → Diagnostik → Anwendung), Diagnostiken als Abnahmekriterien, Begriff der Fehlspezifikation für Sim-to-real.

---

## B. Autofokus: klassische Kriterien und lernbasierte Verfahren

### [7] Ren, Xu, Lam 2018 – Lernbasierter Autofokus in der digitalen Holographie

**Zitat:** Z. Ren, Z. Xu, E. Y. Lam, „Learning-based nonparametric autofocusing for digital holography“, *Optica* 5(4), 337–344 (2018). DOI: [10.1364/OPTICA.5.000337](https://doi.org/10.1364/OPTICA.5.000337).
**Verifikation:** Volltext.
**Was ist es:** Erste Arbeit, die Autofokus in der (sichtbaren) digitalen Holographie als *Regression* mit einem CNN löst (statt Klassifikation über wenige Ebenen).
**Kernidee:** Hologramm → CNN → Abstand (Skalar), ohne Rekonstruktion, ohne Fokuskriterium; Vergleich mit kNN, SVM, MLP auf handgemachten Merkmalen.
**Zahlen:** Datensatz aus 10 Abständen (250–272 mm), je 500 Hologramme → 5000 Hologramme; Split 75/15/10; Adam. CNN-MAE ≈ 0.05 mm (Amplitudenobjekte) bzw. ≈ 0.04–0.06 mm (weitere Experimente, $R^2\approx0.97$), kNN ≈ 1.2–1.6 mm (SVM/MLP ähnlich schlecht). Robust gegenüber Belichtungszeiten (5 ms bis 18 ms).
**Relevanz:** Vorlage für die Regressions-Baseline (WP 3), Beleg, dass ein CNN Fresnel-Streifenskalen direkt lesen kann.
**Übernehmen:** Regressionsformulierung, MAE/$R^2$/Explained-Variance als Metriken, Vergleich mit einfachen Regressoren als Sanity Check. Einschränkung: nur 10 diskrete Abstände; Generalisierung über kontinuierliche $z$ ist in NFH neu.

### [8] Pitkäaho, Manninen, Naughton 2019 – Fokusprädiktion in der DHM

**Zitat:** T. Pitkäaho, A. Manninen, T. J. Naughton, „Focus prediction in digital holographic microscopy using deep convolutional neural networks“, *Applied Optics* 58(5), A202–A208 (2019). DOI: [10.1364/AO.58.00A202](https://doi.org/10.1364/AO.58.00A202).
**Verifikation:** Volltext.
**Was ist es:** CNN-basierte Fokusprädiktion (AlexNet, VGG16 per Transfer Learning) aus Off-axis-DHM-Hologrammen; Vergleich zum Tamura-Kriterium.
**Zahlen:** RMS-Fehler 6.37 mm (AlexNet) bzw. 6.49 mm (VGG16) bei 20 mm akzeptierter Fehlertoleranz (≈100 % bzw. 99.9 % innerhalb); Inferenz 247 ms bzw. 680 ms auf einer Laptop-CPU (Core i5, ohne GPU), gegenüber 932 ms für eine Tamura-Auswertung inkl. Rekonstruktion. Vorverarbeitung: Entfernen von Nullter Ordnung und Twin-Image.
**Relevanz:** Zeigt Laufzeitvorteil auf CPU; aber Genauigkeit weit unter NFH-Anforderungen (mm statt 0.1 mm). Gutes Negativbeispiel für „vortrainierte ImageNet-Netze mit Resize“ – Resize verändert die Streifenskala.
**Übernehmen:** Laufzeitmessung auf CPU als Metrik, Vorsicht bei Transfer Learning mit Größenänderung.

### [9] Pitkäaho, Manninen, Naughton 2017 – Konferenzvorläufer

**Zitat:** T. Pitkäaho, A. Manninen, T. J. Naughton, „Performance of autofocus capability of deep convolutional neural networks in digital holographic microscopy“, in *Digital Holography and Three-Dimensional Imaging*, OSA Technical Digest, paper W2A.5 (2017). DOI: [10.1364/DH.2017.W2A.5](https://doi.org/10.1364/DH.2017.W2A.5).
**Verifikation:** Metadaten über Crossref (Titel/Autoren/Konferenz stimmen; Crossref führt als Erscheinungsjahr 2016, der Digest selbst 2017) und Referenz in [8].
**Relevanz:** Klassifikationsformulierung (wenige Fokusebenen) als Vorläufer von [8]; nur als historischer Verweis zitieren.

### [10] Jaferzadeh, Hwang, Moon, Javidi 2019 – Fokusprädiktion auf Einzelzellebene

**Zitat:** K. Jaferzadeh, S.-H. Hwang, I. Moon, B. Javidi, „No-search focus prediction at the single cell level in digital holographic imaging using deep convolutional neural network“, *Biomedical Optics Express* 10(8), 4276–4289 (2019). DOI: [10.1364/BOE.10.004276](https://doi.org/10.1364/BOE.10.004276).
**Verifikation:** Volltext (PMC).
**Was ist es:** CNN-Regression des Fokusabstands pro Zelle aus Off-axis-Hologrammen (rote Blutkörperchen), ohne Suche über einen Rekonstruktionsstapel.
**Kernidee/Ergebnis:** Fokus pro Objekt statt pro Bild; deutlich schneller als Suchverfahren bei vergleichbarer Genauigkeit; Training auf Hologrammen mit Referenzfokus aus klassischem Kriterium.
**Relevanz:** Beispiel für objektweise Patches und Aggregation; Referenzlabels aus klassischem Verfahren (analog: Referenz-$z_{01}$ aus [2] für Messdaten).

### [11] Montoya et al. 2023 – FocusNET

**Zitat:** M. Montoya, M. J. Lopera, A. Gómez-Ramírez, C. Buitrago-Duque, A. Pabón-Vidal, J. Herrera-Ramirez, J. Garcia-Sucerquia, C. Trujillo, „FocusNET: An autofocusing learning-based model for digital lensless holographic microscopy“, *Optics and Lasers in Engineering* 165, 107546 (2023). DOI: [10.1016/j.optlaseng.2023.107546](https://doi.org/10.1016/j.optlaseng.2023.107546).
**Verifikation:** Volltext der SSRN-Preprintfassung; Journalversion nur Metadaten.
**Was ist es:** CNN-Regression des Rekonstruktionsabstands für *linsenlose* (In-line, Kegelstrahl-)Holographische Mikroskopie – geometrisch der NFH am nächsten unter den optischen Arbeiten.
**Zahlen:** Standardabweichung der Vorhersage ≈ 54 µm; ≈ 600× schneller als Fokussuche über einen Rekonstruktionsstapel; Detektor 3518 × 3518 Pixel.
**Relevanz:** Beleg, dass In-line-Kegelstrahl-Hologramme sub-100-µm-Genauigkeit erlauben; Laufzeitvergleich „gegen Rekonstruktionsstapel“ ist genau die Vergleichslogik gegen [2].
**Übernehmen:** Berichtsformat (Std der Vorhersage, Speedup gegen Suchverfahren), Hinweis auf Patch-basierte Verarbeitung großer Sensoren.

### [12] Rivenson, Wu, Ozcan 2019 – Review: Deep Learning in Holographie

**Zitat:** Y. Rivenson, Y. Wu, A. Ozcan, „Deep learning in holography and coherent imaging“, *Light: Science & Applications* 8, 85 (2019). DOI: [10.1038/s41377-019-0196-0](https://doi.org/10.1038/s41377-019-0196-0).
**Verifikation:** Volltext (Nature/EuropePMC).
**Was ist es:** Übersichtsartikel zu Deep Learning für holographische Rekonstruktion, Phase Retrieval, Autofokus, Auflösungserhöhung, Twin-Image-Unterdrückung.
**Kernaussagen:** Daten­getriebene Verfahren ersetzen iterative Rekonstruktion durch Vorwärtsdurchläufe (Echtzeit), brauchen aber repräsentative Trainingsdaten; Autofokus und Phase Retrieval lassen sich koppeln; Generalisierung über Objektklassen ist das Hauptproblem.
**Relevanz:** Einordnung des Themas im Stand der Forschung (Kapitel 2 der Thesis); Begriffsbildung „learning-based autofocus“.

### [13] Rohou & Grigorieff 2015 – CTFFIND4 (Defokusschätzung aus dem Spektrum)

**Zitat:** A. Rohou, N. Grigorieff, „CTFFIND4: Fast and accurate defocus estimation from electron micrographs“, *Journal of Structural Biology* 192(2), 216–221 (2015). DOI: [10.1016/j.jsb.2015.08.008](https://doi.org/10.1016/j.jsb.2015.08.008).
**Verifikation:** Volltext (PMC).
**Was ist es:** Standardwerkzeug der Kryo-EM zur Schätzung von Defokus und Astigmatismus durch Anpassen einer CTF $\sin(\pi\lambda|g|^2\Delta f - \dots)$ an das Amplitudenspektrum eines Bildes (Thon-Ringe), mit Gütemaß über die Frequenz.
**Relevanz:** Methodische Analogie: In NFH liegen die CTF-Nullstellen bei $|\xi|^2 = n\,\mathrm{Fr}$ – ein nicht-lernbasierter „Spektral-Autofokus“ durch Ringanpassung ist eine naheliegende zusätzliche Baseline und motiviert das Leistungsspektrum als Netzeingabe.
**Übernehmen:** Spektrale Fit-Baseline, Qualitätsmaß als Funktion der Frequenz, Idee der Radialmittelung.

### Klassische Fokuskriterien (VAR, SPEC, GRA, LAP, ToG, GoG)

Die Originalarbeiten zu diesen Kriterien ([2] zitiert dort [1,2,4–8,29]) wurden in dieser Recherche **nicht einzeln verifiziert**. Nutze die Definitionen aus [2] bzw. die Implementierung in `holowizard.core.find_focus.focus_loss_metrics` und zitiere die Primärquellen nur nach eigener Prüfung.

---

## C. NFH-Grundlagen und Phase Retrieval

### [14] Paganin 2006 – Coherent X-Ray Optics

**Zitat:** D. M. Paganin, *Coherent X-Ray Optics*, Oxford University Press (2006). ISBN 978-0-19-856728-8.
**Verifikation:** Metadaten (zitiert in [1], [2] als Quelle des Fresnel-Skalierungstheorems; in Moosmann et al. 2011 als „the cone-beam case relates to the parallel-beam case via a simple rescaling operation“).
**Relevanz:** Standardlehrbuch für Projektionsnäherung, Fresnel-Propagation, Fresnel-Skalierungstheorem, TIE, CTF. Für Kapitel 2 der Thesis als Primärreferenz der Theorie.

### [15] Goodman – Introduction to Fourier Optics

**Zitat:** J. W. Goodman, *Introduction to Fourier Optics*, 4. Aufl., W. H. Freeman (2017) (3. Aufl. Roberts & Company 2005).
**Verifikation:** Metadaten (Vorwort der 4. Auflage gesehen).
**Relevanz:** Fresnel-/Fraunhofer-Näherung, Transferfunktion der Freiraumausbreitung, Abtastbedingungen für numerische Propagation (Transfer-Function- vs. Impulse-Response-Methode). Zitiere für die Abtastbedingung $N \ge 1/\mathrm{Fr}$ (vgl. `01_thesis_erklaerung.md`, Abschnitt 3.5).

### [16] Cloetens et al. 1999 – Holotomographie

**Zitat:** P. Cloetens, W. Ludwig, J. Baruchel, D. Van Dyck, J. Van Landuyt, J. P. Guigay, M. Schlenker, „Holotomography: Quantitative phase tomography with micrometer resolution using hard synchrotron radiation x rays“, *Applied Physics Letters* 75(19), 2912–2914 (1999). DOI: [10.1063/1.125225](https://doi.org/10.1063/1.125225).
**Verifikation:** Abstract/Metadaten.
**Relevanz:** Begründet quantitatives Phase Retrieval aus mehreren Abständen (CTF-Ansatz) und den Begriff Holotomographie; zeigt, dass die Abstände (also $\mathrm{Fr}$) exakt bekannt sein müssen – in der Praxis werden sie dort per Registrierung/Fit kalibriert.

### [17] Zabler et al. 2005 – Optimierung des Phasenkontrasts

**Zitat:** S. Zabler, P. Cloetens, J.-P. Guigay, J. Baruchel, M. Schlenker, „Optimization of phase contrast imaging using hard x rays“, *Review of Scientific Instruments* 76(7), 073705 (2005). DOI: [10.1063/1.1960797](https://doi.org/10.1063/1.1960797).
**Verifikation:** Abstract/Metadaten.
**Relevanz:** Analyse der CTF (Phasen- und Absorptionsanteil, Nullstellen), Wahl der Abstände; Grundlage für die Argumentation, welche Frequenzen bei gegebenem $\mathrm{Fr}$ Information tragen.

### [18] Guigay 1977 und Teague 1983 – CTF und TIE

**Zitate:** J.-P. Guigay, „Fourier transform analysis of Fresnel diffraction patterns and in-line holograms“, *Optik* 49, 121–125 (1977). – M. R. Teague, „Deterministic phase retrieval: a Green's function solution“, *Journal of the Optical Society of America* 73(11), 1434–1441 (1983). DOI: [10.1364/JOSA.73.001434](https://doi.org/10.1364/JOSA.73.001434).
**Verifikation:** Metadaten (über Referenzlisten von Moosmann et al. 2011, Lohse et al. 2020, Zuo et al. 2020).
**Relevanz:** Primärquellen der CTF (Guigay) und der TIE (Teague); in der Thesis als Ursprungszitate.

### [19] Lohse et al. 2020 – HoloTomoToolbox

**Zitat:** L. M. Lohse, A.-L. Robisch, M. Töpperwien, S. Maretzke, M. Krenkel, J. Hagemann, T. Salditt, „A phase-retrieval toolbox for X-ray holography and tomography: software and examples“, *Journal of Synchrotron Radiation* 27, 852–859 (2020). DOI: [10.1107/S1600577520002398](https://doi.org/10.1107/S1600577520002398). Dokumentation: https://irpgoe.gitlab.io/holotomotoolbox/ (Iterative-Seite gesehen: `phaserec_RAAR`, `phaserec_AP`, ...).
**Verifikation:** Volltext.
**Was ist es:** MATLAB-Toolbox der Göttinger Gruppe mit CTF-, TIE-/Paganin-, iterativen (AP/RAAR/Tikhonov) Rekonstruktionen, Hilfsfunktionen (Fresnel-Zahlen, Padding, Flatfield), Beispiele.
**Kernaussage:** Vereinheitlichte Struktur erlaubt Vergleich der Verfahren über Regime hinweg; Fresnel-Zahl $F(A) = A^2/(\lambda z)$ pro Strukturgröße $A$ ordnet Regime (Pixel bis Bildfeld).
**Relevanz:** Zweite Referenzimplementierung neben HoloWizard; Quelle für die Definition verschiedener Fresnel-Zahlen (Pixel- vs. Objektskala) und für Paganin/CTF-Formeln.
**Übernehmen:** Terminologie der Regime, optional unabhängige Gegenprobe der Propagation.

### [20] Huhn, Lohse, Lucht, Salditt 2022 – Nichtlineare Tikhonov-Rekonstruktion

**Zitat:** S. Huhn, L. M. Lohse, J. Lucht, T. Salditt, „Fast algorithms for nonlinear and constrained phase retrieval in near-field X-ray holography based on Tikhonov regularization“, *Optics Express* 30(18), 32871–32886 (2022). DOI: [10.1364/OE.462368](https://doi.org/10.1364/OE.462368). arXiv:2205.01099.
**Verifikation:** Volltext (arXiv).
**Was ist es:** Nichtlineare Verallgemeinerung der CTF-Rekonstruktion als Tikhonov-regularisiertes Problem auf dem vollen Vorwärtsmodell, mit Nebenbedingungen, bei konkurrenzfähiger Rechenzeit (Daten von Wellenleiter-Quellen).
**Relevanz:** Alternative zu ASRM mit ähnlicher Zielsetzung (starke Objekte, Einzelhologramm); zeigt, dass auch dort $\mathrm{Fr}$ als bekannt vorausgesetzt wird.

### [21] Moosmann, Hofmann, Baumbach 2011 – Phase Retrieval bei großen Phasenschüben

**Zitat:** J. Moosmann, R. Hofmann, T. Baumbach, „Single-distance phase retrieval at large phase shifts“, *Optics Express* 19(13), 12066–12073 (2011). DOI: [10.1364/OE.19.012066](https://doi.org/10.1364/OE.19.012066).
**Verifikation:** Volltext.
**Was ist es:** Zwei Wege jenseits der Linearisierung für reine Phasenobjekte (Projektion auf ein effektives lineares Modell im Fourierraum; Entwicklung des Intensitätskontrasts in Potenzen des Abstands).
**Relevanz:** Historischer Kontext für „starke Objekte“ (die Hereon-Gruppe um Moosmann ist Mitautor von [23]); zeigt die Abhängigkeit der Entwicklungen vom Abstand $z$.

### [22] Maretzke & Hohage 2020 – Theorie: Eindeutigkeit und Stabilität

**Zitat:** S. Maretzke, T. Hohage, „Constrained Reconstructions in X-ray Phase Contrast Imaging: Uniqueness, Stability and Algorithms“, in: T. Salditt, A. Egner, D. R. Luke (Hrsg.), *Nanoscale Photonic Imaging*, Topics in Applied Physics 134, Springer (2020), Kap. 14. DOI: [10.1007/978-3-030-34413-9_14](https://doi.org/10.1007/978-3-030-34413-9_14) (Open Access).
**Verifikation:** Volltext.
**Kernaussage:** Unter Supportbedingungen ist das Phasenproblem der XPCI eindeutig lösbar und – im linearisierten Modell – sogar gut gestellt (Stabilitätsabschätzungen); Algorithmen (regularisierte Newton-Verfahren) nutzen das aus.
**Relevanz:** Theoretischer Hintergrund dafür, warum A-priori-Wissen (Support, Nichtnegativität) Rekonstruktionen stabilisiert; ASRM verzichtet bewusst auf Support und ersetzt ihn durch $\Omega_P$ + Regularisierung.

### [23] Yang et al. 2025 – SelfPhish (physikinformiertes GAN, P05/Hereon-Umfeld)

**Zitat:** X. Yang, D. Hailu, V. Kulvait, T. Jentschke, S. Flenner, I. Greving, S. I. Campbell, J. Hagemann, C. G. Schroer, T. M. Wong, J. Moosmann, „Self-supervised physics-informed generative networks for phase retrieval from a single X-ray hologram“, *Optics Express* 33(17), 35832–35851 (2025). DOI: [10.1364/OE.569216](https://doi.org/10.1364/OE.569216). Ergänzendes Material (Supplementary document): figshare 10.6084/m9.figshare.29803343.
**Verifikation:** Volltext; Seitenbereich nach Crossref/Europe PMC (35832–35851), figshare-DOI laut DataCite als „Supplementary document“ (kein Datensatz).
**Was ist es:** Selbstüberwachtes Phase Retrieval (GAN mit eingebautem Fresnel-Vorwärtsmodell) aus einem einzigen Hologramm, ohne gepaarte/simulierte Trainingsdaten; Anwendung auf Hereon/DESY-Daten.
**Relevanz:** Zeigt den Deep-Learning-Stand in der direkten Nachbarschaft deiner Betreuer; auch dort ist $\mathrm{Fr}$ ein Eingabeparameter – ein lernbasierter Autofokus wäre für solche Netze ebenfalls Voraussetzung.

### [24] Ullherr, Diez, Zabler 2022 – Robuste Holotomographie-Rekonstruktion

**Zitat:** M. Ullherr, M. Diez, S. Zabler, „Robust Image Reconstruction Strategy for Multiscalar Holotomography“, *Journal of Imaging* 8(2), 37 (2022). DOI: [10.3390/jimaging8020037](https://doi.org/10.3390/jimaging8020037).
**Verifikation:** Volltext (MDPI, Open Access).
**Relevanz:** Praktische Rekonstruktionsstrategie (Padding-Methoden, Artefaktbehandlung) für Labor-/Synchrotron-Holotomographie; nützlich für die Diskussion von Padding und Randeffekten, nicht zentral.

### [25] Zuo et al. 2020 – TIE-Tutorial

**Zitat:** C. Zuo, J. Li, J. Sun, Y. Fan, J. Zhang, L. Lu, R. Zhang, B. Wang, L. Huang, Q. Chen, „Transport of intensity equation: a tutorial“, *Optics and Lasers in Engineering* 135, 106187 (2020). DOI: [10.1016/j.optlaseng.2020.106187](https://doi.org/10.1016/j.optlaseng.2020.106187).
**Verifikation:** Abstract/Metadaten (OSTI-Eintrag).
**Relevanz:** Ausführliche, gut lesbare Herleitung der TIE, ihrer Lösungen und Grenzen – Einstieg für Kapitel 2, wenn Teague 1983 zu knapp ist.

### [26] Langer, Cloetens, Guigay, Peyrin 2008 – Vergleich direkter Verfahren

**Zitat:** M. Langer, P. Cloetens, J.-P. Guigay, F. Peyrin, „Quantitative comparison of direct phase retrieval algorithms in in-line phase tomography“, *Medical Physics* 35(10), 4556–4566 (2008). DOI: [10.1118/1.2975224](https://doi.org/10.1118/1.2975224).
**Verifikation:** Metadaten (Wiley-Seite ohne Volltext).
**Relevanz:** Systematischer Vergleich CTF/TIE/gemischt; Vorlage für die Struktur eines quantitativen Methodenvergleichs.

### [27] Hagemann 2017 – Dissertation: NFH jenseits idealisierter Probe-Annahmen

**Zitat:** J. Hagemann, *X-Ray Near-Field Holography: Beyond Idealized Assumptions of the Probe*, Göttingen Series in X-ray Physics, Universitätsverlag Göttingen (2017). DOI: [10.17875/gup2017-1055](https://doi.org/10.17875/gup2017-1055).
**Verifikation:** Volltext (PDF-Textfassung teilweise fehlerhaft extrahiert).
**Relevanz:** Monographie eines Betreuer-Koautors zur Rolle der nicht-idealen Beleuchtung (Probe) in der NFH – wichtig für die Sim-to-real-Diskussion (Probe-Struktur, Flatfield-Korrektur, Wellenleiter vs. FZP) und als Nachschlagewerk für die Göttinger/DESY-Terminologie.

---

## D. Simulationsbasierte Inferenz und Unsicherheit

### [28] Boelts et al. 2025 – sbi reloaded

**Zitat:** J. Boelts, M. Deistler, M. Gloeckler, Á. Tejero-Cantero, J.-M. Lueckmann, G. Moss, P. Steinbach, T. Moreau, F. Muratore, J. Linhart, C. Durkan, J. Vetter, B. K. Miller, M. Herold, A. Ziaeemehr, M. Pals, T. Gruner, S. Bischoff, N. Krouglova, R. Gao, J. K. Lappalainen, B. Mucsányi, F. Pei, A. Schulz, Z. Stefanidi, P. Rodrigues, C. Schröder, F. Abu Zaid, J. Beck, J. Kapoor, D. S. Greenberg, P. J. Gonçalves, J. H. Macke, „sbi reloaded: a toolkit for simulation-based inference workflows“, *Journal of Open Source Software* 10(108), 7754 (2025). DOI: [10.21105/joss.07754](https://doi.org/10.21105/joss.07754). Code: https://github.com/sbi-dev/sbi.
**Verifikation:** Volltext (JOSS-Paper und arXiv-Fassung).
**Was ist es:** Beschreibung der aktuellen `sbi`-Bibliothek (PyTorch): NPE/NLE/NRE, Flows/Flow-Matching/Diffusion, Embedding-Netze, Diagnostik (SBC, Expected Coverage, TARP, L-C2ST), Mehrrunden-Inferenz, Zusammenfassende Statistiken.
**Relevanz:** Werkzeug für WP 4; Autorenkreis identisch mit [6].
**Übernehmen:** API-Muster `NPE(prior, embedding_net=...)`, Diagnostikfunktionen.

### [29] Tejero-Cantero et al. 2020 – sbi (erste Version)

**Zitat:** Á. Tejero-Cantero, J. Boelts, M. Deistler, J.-M. Lueckmann, C. Durkan, P. J. Gonçalves, D. S. Greenberg, J. H. Macke, „sbi: A toolkit for simulation-based inference“, *Journal of Open Source Software* 5(52), 2505 (2020). DOI: [10.21105/joss.02505](https://doi.org/10.21105/joss.02505).
**Verifikation:** Metadaten (zitiert in [28]).

### [30] Cranmer, Brehmer, Louppe 2020 – The frontier of simulation-based inference

**Zitat:** K. Cranmer, J. Brehmer, G. Louppe, „The frontier of simulation-based inference“, *PNAS* 117(48), 30055–30062 (2020). DOI: [10.1073/pnas.1912789117](https://doi.org/10.1073/pnas.1912789117).
**Verifikation:** Metadaten (mehrfach in [6] zitiert).
**Relevanz:** Übersicht über SBI-Familien (ABC, NPE, NLE, NRE) und aktive Lernstrategien; Standard-Einstiegszitat.

### [31] Grundlagenarbeiten zu NPE und Diagnostik (alle: Metadaten, zitiert in [6]/[28])

- G. Papamakarios, I. Murray, „Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation“, NeurIPS 2016. arXiv:1605.06376. – Ursprung der NPE-Idee (Mixture Density Networks als Posteriorschätzer).
- D. S. Greenberg, M. Nonnenmacher, J. H. Macke, „Automatic Posterior Transformation for Likelihood-Free Inference“, ICML 2019 (PMLR 97). arXiv:1905.07488. – APT/SNPE-C, Standardverfahren in `sbi`.
- J.-M. Lueckmann, J. Boelts, D. S. Greenberg, P. J. Gonçalves, J. H. Macke, „Benchmarking Simulation-Based Inference“, AISTATS 2021 (PMLR 130). arXiv:2101.04653. – Benchmark, Metriken (C2ST).
- S. Talts, M. Betancourt, D. Simpson, A. Vehtari, A. Gelman, „Validating Bayesian Inference Algorithms with Simulation-Based Calibration“, arXiv:1804.06788 (2018). – SBC.
- J. Hermans, A. Delaunoy, F. Rozet, A. Wehenkel, V. Begy, G. Louppe, „A Trust Crisis In Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful“, *TMLR* (2022). arXiv:2110.06581. – Überkonfidenz von SBI-Posterioren, Expected Coverage, Ensembles.
- P. Lemos, A. Coogan, Y. Hezaveh, L. Perreault-Levasseur, „Sampling-Based Accuracy Testing of Posterior Estimators for General Inference“, ICML 2023 (PMLR 202). arXiv:2302.03026. – TARP.
- J. Linhart, A. Gramfort, P. L. C. Rodrigues, „L-C2ST: Local Diagnostics for Posterior Approximations in Simulation-Based Inference“, NeurIPS 2023. arXiv:2306.03580. – lokale Kalibrierungstests.
**Relevanz:** Primärzitate für die in WP 4/5 verwendeten Diagnostiken; vor Zitierung in der Thesis jeweils Abstract prüfen.

### [32] Kendall & Gal 2017 – Aleatorische und epistemische Unsicherheit

**Zitat:** A. Kendall, Y. Gal, „What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?“, NeurIPS 2017. arXiv:1703.04977.
**Verifikation:** Volltext (arXiv).
**Kernidee:** Heteroskedastische Regression (Netz sagt $\mu$ und $\log\sigma^2$ voraus, Gauß-NLL) erfasst datenabhängiges Rauschen (aleatorisch); MC-Dropout/Bayessche Gewichte erfassen Modellunsicherheit (epistemisch); Kombination beider.
**Relevanz:** Einfachste Unsicherheits-Baseline für WP 3/4 vor dem Einsatz von NPE.

### [33] Lakshminarayanan, Pritzel, Blundell 2017 – Deep Ensembles

**Zitat:** B. Lakshminarayanan, A. Pritzel, C. Blundell, „Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles“, NeurIPS 2017. arXiv:1612.01474.
**Verifikation:** Volltext.
**Kernidee:** Mehrere unabhängig initialisierte Netze mit proper scoring rule (NLL) trainieren; Mischung der Vorhersagen liefert gut kalibrierte Unsicherheit, robust unter Verteilungsverschiebung; einfacher als Bayessche Netze.
**Relevanz:** Robuste Unsicherheitsbaseline und Mittel gegen Überkonfidenz (auch für NPE-Ensembles empfohlen in [6]).

### [34] Gal & Ghahramani 2016 – MC-Dropout; Guo et al. 2017 – Kalibrierung

**Zitate:** Y. Gal, Z. Ghahramani, „Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning“, ICML 2016 (PMLR 48). arXiv:1506.02142. – C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, „On Calibration of Modern Neural Networks“, ICML 2017 (PMLR 70). arXiv:1706.04599.
**Verifikation:** Metadaten (zitiert in [32], [33]).
**Relevanz:** MC-Dropout als billige epistemische Unsicherheit; Kalibrierungsbegriff (Reliability-Diagramme, Temperature Scaling) für die Klassifikationsvariante.

### [35] Tobin et al. 2017 – Domain Randomization

**Zitat:** J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel, „Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World“, IEEE/RSJ IROS 2017. DOI: [10.1109/IROS.2017.8202133](https://doi.org/10.1109/IROS.2017.8202133). arXiv:1703.06907.
**Verifikation:** Volltext (arXiv).
**Kernidee:** Starke Randomisierung nicht-essentieller Simulationsparameter (Texturen, Beleuchtung, Rauschen) zwingt das Netz, auf die invarianten Merkmale zu achten; Transfer auf Realdaten ohne Realtraining.
**Relevanz:** Begründung der Randomisierung von Phantomen, Probes, Rauschen und Flatfields in HoloForge (WP 2) und Diskussion der Sim-to-real-Lücke (WP 6).

---

## E. Software und Daten

| Nr. | Ressource | Angabe | Verifikation | Nutzen |
|---|---|---|---|---|
| [5] | HoloWizard 3.0.6 (Core, Forge, Livereco, Pipe) | Zenodo 10.5281/zenodo.8349364; lokal `/tmp/holo311` | Code vollständig gelesen | Simulator, Baseline, Integrationsziel |
| [28] | `sbi` (sbi-dev/sbi) | https://github.com/sbi-dev/sbi, Doku https://sbi.readthedocs.io | Paper Volltext | NPE, Diagnostik |
| [19] | HoloTomoToolbox (MATLAB) | https://irpgoe.gitlab.io/holotomotoolbox/ | Doku-Seite gesehen | unabhängige Gegenprobe (optional) |
| [36] | xraylib | T. Schoonjans, A. Brunetti, B. Golosio, M. Sanchez del Rio, V. A. Solé, C. Ferrero, L. Vincze, „The xraylib library for X-ray–matter interactions. Recent developments“, *Spectrochimica Acta Part B* 66(11–12), 776–784 (2011). DOI: 10.1016/j.sab.2011.09.011 | Metadaten (nicht separat geprüft; Bibliothek lokal genutzt) | $\delta,\beta$ der Phantome |
| – | PyTorch, TensorBoard, h5py, scipy (Nelder–Mead) | – | – | Training, Tracking, Daten, Optimierer |
| – | Kaggle Notebooks | 30 GPU-Stunden/Woche (P100 bzw. 2×T4), 12 h Sitzungslimit, laut Kaggle-Doku | Doku gelesen | Fallback-GPU für kleine Experimente |
| – | Google Colab (Free) | GPU-Zugang „heavily restricted“, Limits dynamisch, laut Colab-FAQ | Doku gelesen | nur Notfall |
| – | P05-Messdaten (Spinnenhaar, Zahn, Kaktusnadel, Mg-Drähte; Beamtime-IDs 11010216, 11014415, 11008588, 11018168 laut [2]) | Zugang über Johannes Dora/DESY | Metadaten aus [2] | Testdaten mit Referenz-$z_{01}$ |

---

## Empfohlene Lesereihenfolge

| Reihenfolge | Quelle | Warum jetzt | Fragen, die du beim Lesen beantworten solltest |
|---|---|---|---|
| 1 | `01_thesis_erklaerung.md` (dieses Repo) | Gesamtbild, Notation, Zahlen | Kannst du $\mathrm{Fr}$ für $z_{01}=120$ mm im Kopf abschätzen? Warum ist $z_{02}$ unkritisch? |
| 2 | [2] Dora 2025 | Baseline und Problemformulierung; kurz, gut lesbar | Was genau ist der MFE im Code (`loss_se_all[-1]`)? Welche vier Anforderungen an Fokuskurven? Wie viele Rekonstruktionen braucht NM und was kostet eine? Welche Werte in Tab. 2 kannst du mit `calc_Fr` reproduzieren? |
| 3 | HoloWizard-Code: `core/find_focus/find_focus_z01.py`, `core/models/cone_beam.py`, `forge/utils/utilities.py`, `forge/experiment/simulation/nfh_simulation.py`, `core/scripts/examples/find_focus/magnesium_wire.py` | Verbindet Paper und Praxis | Welche Einheiten? Wo wird $\sqrt I$ gebildet? Was ist `a0`? Warum Padding 4? Was gibt Forge zurück – Amplitude oder Intensität? |
| 4 | [1] Dora 2024 (Abschnitte 2–4) | Versteht den inneren Löser und $\Omega_P$ | Welche Vorverarbeitung macht ASRM robust? Was bedeuten die Multigrid-Parameter in Tab. 3 von [2]? |
| 5 | [14] Paganin 2006, Kap. zu Fresnel-Propagation/Skalierungstheorem, oder [15] Goodman Kap. 4–5 | Theoriefundament | Herleitung von $\Delta x/M$, $z_{12}/M$; Abtastbedingung der Transferfunktion |
| 6 | [3] Paganin 2002, [19] Lohse 2020, [17] Zabler 2005 | CTF/TIE-Regime, Fresnel-Zahlen pro Skala | Wo liegen die CTF-Nullstellen in Pixelfrequenzen? Für welche Objekte gilt die Linearisierung nicht? |
| 7 | [4] Fienup 1982 | Projektionsalgorithmen | Warum stagniert ER? Was macht HIO anders? Wo ist die Verbindung zu PGD/ASRM? |
| 8 | [7] Ren 2018, [11] FocusNET 2023, [8] Pitkäaho 2019, [12] Rivenson 2019 | Stand der lernbasierten Autofokussierung | Welche Zielgröße, welcher Fehler, welche Datenmenge? Warum ist Resize gefährlich? Was fehlt gegenüber NFH (Kegelstrahl, starke Objekte, kontinuierliches $z$)? |
| 9 | [13] CTFFIND4 | Spektrale Baseline | Wie wird ein Ringmuster an ein Spektrum gefittet? Übertragbar auf $\lvert\xi\rvert^2 = n\mathrm{Fr}$? |
| 10 | [6] SBI-Guide (Kap. Workflow + Diagnostik), dann [28] sbi reloaded | Probabilistische Variante | Welcher Prior für $z_{01}$? Wie viele Simulationen? Welche Diagnostiken sind Abnahmekriterien? Wie erkennst du Fehlspezifikation auf Messdaten? |
| 11 | [32], [33], [35] | Einfache Unsicherheits-Baselines, Sim-to-real | Was trennt aleatorisch/epistemisch im Autofokus? Welche Simulationsparameter randomisierst du? |
| 12 | [20], [22], [23], [27] | Vertiefung, Diskussionskapitel | Welche Annahmen über die Probe sind in deinen Simulationen idealisiert? |

---

## Verworfene Kandidaten und Gründe

| Kandidat | Grund |
|---|---|
| Primärquellen der klassischen Fokuskriterien (Langehanenberg/Memmolo/Dubois u. a., in [2] als [1,2,4–8,29] zitiert) | in dieser Recherche nicht einzeln verifiziert; nur über [2] referenzieren |
| Arbeiten der Betreuergruppe zu ML in der Röntgenbildgebung (u. a. ICASSP-2026-Einträge in Google Scholar) | nur Trefferlisten-Metadaten ohne prüfbaren Volltext/DOI; nicht aufgenommen |
| Zayko, Kfir, Ropers, „Polarization-Sensitive Coherent Diffractive Imaging Using HHG“ (Kap. 18 in *Nanoscale Photonic Imaging*) | Suchtreffer ohne thematischen Bezug (Fernfeld-CDI mit HHG) |
| Fienup 1987, „Reconstruction of a complex-valued object from the modulus of its Fourier transform using a support constraint“, JOSA A 4(1), 118–123 | thematisch Fernfeld/Support; für NFH-Autofokus nicht nötig (ggf. Fußnote zu Supportbedingungen) |
| Allgemeine Salditt-Übersichtsartikel zur NFH | nicht im Volltext geprüft; stattdessen [19], [20], [22], [27] verwendet |
| Google-Colab-/Kaggle-Dokumentation als „Literatur“ | nur als Ressourcenhinweis (Abschnitt E), nicht zitierfähig |
