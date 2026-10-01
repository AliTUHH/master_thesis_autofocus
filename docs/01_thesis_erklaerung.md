# Lernbasierter Autofokus für Röntgen-Nahfeldholographie – Erklärung des Thesis-Themas

Stand: Oktober 2026. Grundlage: Ausschreibung „Learning-based autofocus for Holography“ (TUHH, Betreuung Daniel Hernández Durán, Johannes Dora, Tobias Knopp), die Pflichtliteratur [1]–[6] (Nummerierung wie in der Ausschreibung, Vollzitate in `02_literatur.md`) und der Quellcode von HoloWizard 3.0.6 (Core, Forge, Livereco). Alle Zahlenwerte in diesem Dokument wurden mit `holowizard.forge.utils.calc_Fr`, `holowizard.core.models.cone_beam.ConeBeam` bzw. `xraylib` nachgerechnet. Einheiten wie in HoloWizard: Längen in mm, Pixelgröße in mm, Energie in keV, Wellenlänge in nm.

---

## 1. Das Thema in drei Sätzen

In der Röntgen-Nahfeldholographie (NFH) an der Nano-Imaging-Endstation P05 (PETRA III, DESY) wird aus einem einzigen Intensitätsbild (Hologramm) der komplexe projizierte Brechungsindex einer Probe rekonstruiert; dafür muss das Vorwärtsmodell exakt parametrisiert sein, und der einzige kritische, nicht präzise messbare Parameter ist der Fokus-Objekt-Abstand $z_{01}$, der die Fresnel-Zahl $\mathrm{Fr}$ festlegt. Der aktuelle Stand der Technik [2] bestimmt $z_{01}$ durch eine geschachtelte Optimierung (Nelder–Mead über $z_{01}$, in jedem Probepunkt eine vollständige ASRM-Rekonstruktion [1]), was robust ist, aber Minuten pro Datensatz dauert. Deine Thesis untersucht, ob ein neuronales Netz $\mathrm{Fr}$ bzw. $z_{01}$ (und eine kalibrierte Unsicherheit dafür, z. B. per simulationsbasierter Inferenz [6]) direkt aus dem Hologramm in Millisekunden schätzen kann – trainiert auf synthetischen HoloForge-Daten [5] und getestet auf Messdaten – und wie eine solche Schätzung den modellbasierten Autofokus ersetzen oder beschleunigen kann.

---

## 2. Physikalischer Hintergrund

### 2.1 Synchrotronstrahlung und Kohärenz

- Hartes Röntgenlicht (hier $E = 11\,\mathrm{keV}$ bzw. $17\,\mathrm{keV}$ in [2]) hat die Wellenlänge
  $$\lambda = \frac{hc}{E} \approx \frac{1.2398\ \mathrm{keV\,nm}}{E}, \qquad \lambda(11\,\mathrm{keV}) = 0.11271\ \mathrm{nm}.$$
  HoloWizard verwendet exakt diese Konstante (`lam = 1.2398 / energy` in nm).
- Propagationsbasierter Phasenkontrast braucht *räumliche Kohärenz*: Die Wellenfronten aus benachbarten Punkten des Objekts müssen in der Detektorebene interferieren können. An einer Undulator-Beamline der dritten Generation wie P05 ist die transversale Kohärenzlänge groß genug, damit sich Fresnel-Interferenzstreifen über viele Pixel ausbilden. Zeitliche Kohärenz (Monochromator, $\Delta E/E \sim 10^{-4}$ bis $10^{-2}$) begrenzt, wie viele Streifenordnungen sichtbar bleiben.
- Bei P05 wird der Strahl mit einer Fresnel-Zonenplatte (FZP) fokussiert; der Fokus wirkt als (nahezu) punktförmige Quelle, aus der ein *Kegelstrahl* divergiert (siehe Abschnitt 4). Alle Abstände in [1], [2] und HoloWizard werden vom Fokus aus gemessen.

### 2.2 Brechungsindex für harte Röntgenstrahlung

Der komplexe Brechungsindex wird als
$$n = 1 - \delta + i\beta$$
geschrieben. $\delta$ beschreibt die Phasenverschiebung (Dispersion), $\beta$ die Absorption. Für harte Röntgenstrahlung gilt $\delta, \beta \ll 1$ und meist $\delta \gg \beta$. Mit `xraylib` (das auch HoloForge intern verwendet) ergeben sich bei 11 keV:

| Material | $\rho$ / g cm$^{-3}$ | $\delta$ | $\beta$ | $\delta/\beta$ | Phase pro µm $-k\delta t$ | Phase bei 30 µm |
|---|---|---|---|---|---|---|
| Mg | 1.74 | $2.97\cdot10^{-6}$ | $2.48\cdot10^{-8}$ | 120 | $-0.166$ rad | $-4.97$ rad |
| Al | 2.70 | $4.51\cdot10^{-6}$ | $4.80\cdot10^{-8}$ | 94 | $-0.252$ rad | $-7.6$ rad |
| Fe | 7.87 | $1.27\cdot10^{-5}$ | $9.34\cdot10^{-7}$ | 13.5 | $-0.705$ rad | $-21$ rad |
| Cu | 8.96 | $1.38\cdot10^{-5}$ | $1.37\cdot10^{-6}$ | 10 | $-0.768$ rad | $-23$ rad |
| Au | 19.32 | $2.41\cdot10^{-5}$ | $1.61\cdot10^{-6}$ | 15 | $-1.34$ rad | $-40$ rad |
| H$_2$O | 1.00 | $1.91\cdot10^{-6}$ | $3.6\cdot10^{-9}$ | 530 | $-0.107$ rad | $-3.2$ rad |

Lies daraus ab:
- Ein 30 µm dicker Magnesiumdraht (Standardphantom in HoloForge, Material der Drähte in [2]) erzeugt rund $-5$ rad Phasenschub und nur 8 % Intensitätsabsorption: ein typisches *stark phasenschiebendes, schwach absorbierendes* Objekt. Die Phasenschübe liegen jenseits von $\pi$, also außerhalb der Gültigkeit linearisierter Verfahren (CTF, schwache Objekte).
- $\delta/\beta$ ist materialabhängig (10 bis >1000). Einmaterial-Verfahren wie Paganin [3] setzen ein bekanntes, konstantes $\delta/\beta$ voraus.
- Die Wellenzahl ist $k = 2\pi/\lambda = 5.57\cdot10^{10}\ \mathrm{m^{-1}} = 55.7\ \mathrm{µm^{-1}}$.

### 2.3 Projektionsnäherung und Austrittswelle

Unter der *Projektionsnäherung* (Objekt dünn genug, dass Beugung innerhalb des Objekts vernachlässigbar ist) wird das Objekt durch den komplexen projizierten Brechungsindex beschrieben ([1], [2] Gl. (1)):
$$\tilde O(x,y) := -k\int_0^d \big(\delta(x,y,z) - i\beta(x,y,z)\big)\,dz = \varphi(x,y) + i\mu(x,y),$$
mit Phasenschub $\varphi \le 0$ und Absorptionsterm $\mu \ge 0$. Die Austrittswelle hinter dem Objekt ist
$$\psi_\mathrm{exit} = \exp\!\big(i\tilde O\big)\cdot P = \exp(i\varphi)\,\exp(-\mu)\cdot P,$$
wobei $P$ die einfallende Beleuchtung (Probe) ist. Das Betragsquadrat $|\psi_\mathrm{exit}|^2 = e^{-2\mu}|P|^2$ ist das Lambert–Beer-Gesetz mit linearem Schwächungskoeffizienten $\mu_\mathrm{lin} = 4\pi\beta/\lambda = 2k\beta$, d. h. $\mu = k\int\beta\,dz$ und $2\mu = \mu_\mathrm{lin} t$.

Physikalische Nebenbedingungen (in [1], [2] als Menge $\Omega_P$ formuliert): $\operatorname{Re}\tilde O = \varphi \le 0$ (Röntgenlicht wird nur negativ phasenverschoben, Elektronendichte nicht negativ) und $\operatorname{Im}\tilde O = \mu \ge 0$ (Objekt fügt keine Intensität hinzu). Diese Nebenbedingungen sind das zentrale A-priori-Wissen von ASRM und zugleich der Hebel des MFE-Kriteriums in [2] (Abschnitt 6.3).

Hinweis zur HoloForge-Implementierung: `PhantomGenerator._create_physical_properties` setzt `phaseshift = -k*delta*t` und `absorption = 2*beta*k*t` und bildet die Austrittswelle als `exp(1j*(phaseshift + 1j*absorption))`. Damit ist die *Amplitude* mit $e^{-2k\beta t}$ gedämpft, während nach Gl. (1) in [1]/[2] $\mu = k\beta t$ gilt. Nach meiner Lesart ist die Absorption in Forge-Phantomen also um den Faktor 2 im Exponenten stärker als physikalisch bei gegebener Dicke (oder die Dicke entspricht der doppelten nominalen Dicke). Für den Autofokus ist das unkritisch (es ändert nur das effektive $\delta/\beta$ der Phantome), für Aussagen zu realistischen Materialkontrasten solltest du es mit Johannes Dora klären.

### 2.4 Probe, Flatfield und Detektor

- Die reale Beleuchtung ist nicht eben: Die FZP-Optik erzeugt eine strukturierte Probe $P = A_0 e^{i\phi_0}$ ([1] Gl. (4)). Ein *Flatfield* (Leeraufnahme ohne Objekt) misst $|P|^2$ in der Detektorebene; die Division Hologramm/Flatfield („Flatfield-Korrektur“) entfernt die Intensitätsstruktur näherungsweise, nicht aber die Phasenstruktur der Probe. HoloWizard Core bietet zusätzlich eine komponentenbasierte Flatfield-Korrektur (`find_focus_flatfieldcorrection.py`, Korrektur im Log-Bereich mit einem gepickelten Komponentenmodell).
- Der Detektor misst nur die *Intensität* $I_\mathrm{det} = |\mathcal D(\psi_\mathrm{exit})|^2$. Bei P05 ([2] Abschnitt 3.6): Szintillator (10 µm Gadox) + sCMOS (Hamamatsu C12849-101U), 6.5 µm Pixel, 2048 × 2048 Pixel, 16 bit, Belichtungszeiten 0.8–1.5 s.
- Rauschen: Photonenrauschen (Poisson), Ausleserauschen (gaußisch), Szintillator-Unschärfe, Flatfield-Restfehler. HoloForge modelliert additiv gaußisches oder Poisson-Rauschen auf der Hologramm-Amplitude sowie optional Rauschen und polynomiale Gradienten auf der Probe (`forge/utils/noise.py`, `forge/experiment/probe/polynomial_probe.py`).

---

## 3. Fresnel-Propagation und Fresnel-Zahl

### 3.1 Der Fresnel-Propagator

Die Freiraumausbreitung einer paraxialen Welle über die Distanz $z$ ist im Fourierraum eine Multiplikation mit dem Fresnel-Kern. Mit physikalischen Ortsfrequenzen $f_x, f_y$ (Zyklen pro Länge):
$$\mathcal D_z(\psi) = \mathcal F^{-1}\Big[\exp\!\big(-i\pi\lambda z\,(f_x^2+f_y^2)\big)\cdot\mathcal F[\psi]\Big]$$
(der globale Phasenfaktor $e^{ikz}$ ist weggelassen; die Vorzeichenkonvention von $i$ folgt HoloWizard). Rechnet man in *Pixeleinheiten* – Frequenzen $\xi,\eta$ in Zyklen pro Pixel, $f_x = \xi/\Delta x$ – wird daraus die Form, die HoloWizard und [1], [2] Gl. (3) benutzen:
$$\mathcal D_\mathrm{Fr}(\psi) = \mathcal F^{-1}\Big[\exp\!\Big(-\frac{i\pi}{\mathrm{Fr}}(\xi^2+\eta^2)\Big)\cdot\mathcal F[\psi]\Big], \qquad \mathrm{Fr} = \frac{\Delta x^2}{\lambda z}.$$
Im Code: `fresnel_propagator_torch.py` baut den Kern aus `torch.fft.fftfreq`-Gittern als `exp(-1j*pi/Fr*(xi²+eta²))`; Rückpropagation entspricht $\mathrm{Fr}\to-\mathrm{Fr}$. In HoloForge erzeugt `NFHSetup.create_kernel(Fr)` denselben Kern auf dem gepaddeten Gitter (`fftfreq(probe_size)`), und `NFHSimulation` rechnet `psi_det = ifft2(fft2(probe*exp(1j*obj)) * kernel)`.

Zentrale Konsequenz: In Pixeleinheiten hängt das Hologramm nur noch von $\tilde O$ (auf dem Pixelgitter), der Probe $P$ und dem *einen* Skalar $\mathrm{Fr}$ ab. Ein Netz, das Hologramme auf dem Pixelgitter sieht, kann deshalb prinzipiell nur $\mathrm{Fr}$ lernen; $z_{01}$ folgt daraus über die Geometrie (Abschnitt 4.6).

### 3.2 Fresnel-Zahl: allgemeine Definition vs. Pixel-Fresnel-Zahl

- Allgemein: Für eine Struktur der Größe $a$ (Apertur, Objektmerkmal) ist $F = a^2/(\lambda z)$. $F \gg 1$: geometrische Optik/Schatten; $F \sim 1$: Fresnel-Nahfeld; $F \ll 1$: Fraunhofer-Fernfeld.
- Pixel-Fresnel-Zahl (HoloWizard, [1], [2]): $a = \Delta x$ (effektive Pixelgröße). Typische Werte in NFH liegen bei $10^{-5}$ bis $10^{-3}$. Ein einzelnes Pixel ist für sich betrachtet also tief im „Fernfeld“ – aber ein Merkmal aus $N$ Pixeln hat $F = N^2\,\mathrm{Fr}$. Die Grenze $F = 1$ liegt bei
  $$N_\ast = \frac{1}{\sqrt{\mathrm{Fr}}}\quad\text{Pixel} \qquad (\mathrm{Fr} = 2.37\cdot10^{-4}: N_\ast \approx 65\ \text{px}).$$
  Strukturen größer als $N_\ast$ Pixel werden holographisch (Nahfeld, Kanten mit Fresnel-Streifen) abgebildet, kleinere Strukturen „verschmieren“ über ihre Beugungsscheibe. Weil ganze Objekte (hunderte bis tausende Pixel) weit über $N_\ast$ liegen, spricht man von *Nahfeld*holographie.

### 3.3 Warum „Hologramm“?

Das Intensitätsbild ist die Interferenz der ungestörten Beleuchtung (Referenzwelle) mit der am Objekt gestreuten Welle auf derselben Achse – ein In-line- bzw. Gabor-Hologramm. Die Phaseninformation ist in Lage und Kontrast der Interferenzstreifen kodiert, nicht verloren, sondern „verschlüsselt“: Deshalb kann man sie durch Rückpropagation plus Nebenbedingungen zurückgewinnen (Phase Retrieval, Abschnitt 5). Ohne Propagation ($z \to 0$, $\mathrm{Fr}\to\infty$) sähe man nur die Absorption $e^{-2\mu}|P|^2$.

### 3.4 Streifenskala und CTF-Nullstellen

Die charakteristische Streifenbreite der Fresnel-Beugung ist $\sqrt{\lambda z_\mathrm{eff}}$; in Pixeln ausgedrückt
$$\frac{\sqrt{\lambda z_\mathrm{eff}}}{\Delta x_\mathrm{eff}} = \frac{1}{\sqrt{\mathrm{Fr}}} = N_\ast .$$
Für schwache Objekte hat die Kontrastübertragungsfunktion (Abschnitt 5.2) Nullstellen des Phasenkontrasts bei $\pi\lambda z f^2 = n\pi$, also in Pixelfrequenzen bei
$$\xi_n^2 + \eta_n^2 = n\cdot\mathrm{Fr}, \qquad n = 1,2,\dots$$
Im Leistungsspektrum eines Hologramms erscheinen konzentrische Ringe, deren Radien-Quadrate äquidistant mit Abstand $\mathrm{Fr}$ liegen. Das ist die direkteste „Signatur“ der Fresnel-Zahl im Bild und das Argument für spektrale Eingaberepräsentationen (Abschnitt 7.5) – analog zur Defokusschätzung aus Thon-Ringen in der Kryo-EM (CTFFIND4).

Zahlen für die Standardgeometrie (11 keV, $z_{02} = 20\,000$ mm, $\Delta x = 6.5$ µm):

| $z_{01}$ / mm | $\mathrm{Fr}$ | $\sqrt{\lambda z_\mathrm{eff}}$ | $N_\ast = 1/\sqrt{\mathrm{Fr}}$ / px | 1. CTF-Nullstelle $\xi_1 = \sqrt{\mathrm{Fr}}$ / Zyklen px$^{-1}$ |
|---|---|---|---|---|
| 20 | $1.876\cdot10^{-5}$ | 1.50 µm | 231 | 0.0043 |
| 50 | $4.697\cdot10^{-5}$ | 2.37 µm | 146 | 0.0069 |
| 100 | $9.419\cdot10^{-5}$ | 3.35 µm | 103 | 0.0097 |
| 250 | $2.373\cdot10^{-4}$ | 5.27 µm | 65 | 0.0154 |
| 500 | $4.806\cdot10^{-4}$ | 7.41 µm | 46 | 0.0219 |

Auflösbarkeit der Ringe: Zwei benachbarte Nullstellen bei $\xi$ haben den Abstand $\Delta\xi \approx \mathrm{Fr}/(2\xi)$; mit der Frequenzauflösung $1/N$ eines $N$-Pixel-Bildes sind sie nur bis $\xi < N\,\mathrm{Fr}/2$ getrennt. Bei $N = 2048$ und $\mathrm{Fr} = 2.4\cdot10^{-4}$ ist das $\xi \lesssim 0.24$ (fast das ganze Band); bei $N = 256$ nur $\xi \lesssim 0.03$, also nur die ersten etwa vier Ringe. Kleine Bildausschnitte tragen also deutlich weniger $\mathrm{Fr}$-Information.

### 3.5 Abtastbedingung und Padding

Der Fresnel-Kern $\exp(-i\pi(\xi^2+\eta^2)/\mathrm{Fr})$ wird auf einem Gitter mit Frequenzschritt $1/N$ abgetastet. Aliasfreiheit verlangt, dass die Kernphase zwischen Nachbarfrequenzen um höchstens $\pi$ wächst; bei der Nyquist-Frequenz $\xi = 1/2$ ergibt das
$$N \ \ge\ \frac{1}{\mathrm{Fr}}.$$
Äquivalent: Die Impulsantwort (Chirp) der Propagation erstreckt sich über etwa $1/(2\mathrm{Fr})$ Pixel; ist das Gitter kleiner, wickeln sich die Streifen zyklisch um („wrap-around“). Das erklärt das Padding in HoloWizard (Beispielskripte: `padding_factor = 4`, 2048 px → 8192 px) und die zwei Gittergrößen in [2] Tab. 1: Für $\mathrm{Fr} \approx 3\text{–}5\cdot10^{-4}$ ist $1/\mathrm{Fr} \approx 2000\text{–}3200 < 8192$; für $\mathrm{Fr} = 7.8\cdot10^{-5}$ bzw. $1.2\cdot10^{-4}$ ist $1/\mathrm{Fr} \approx 12\,800$ bzw. $8100$, daher 14 336 px.

Praktische Folge für deine Simulationen: Ein 256-px-Detektor mit 6.5 µm Pixeln und Padding 2 (Gitter 512) bei $\mathrm{Fr} \in [4.7\cdot10^{-5}, 2.8\cdot10^{-4}]$ ($1/\mathrm{Fr} \in [3600, 21\,000]$) würde die Bedingung deutlich verletzen; solche Daten taugten nur als Smoke-Test der Pipeline. `configs/data_small.yaml` umgeht das, indem der 2048-px-Detektor um Faktor 8 gebinnt wird (`downsample_factor: 8` → 256 px mit effektiver Pixelgröße 52 µm, `padding_factor: 2` → Gitter 512): $\mathrm{Fr}$ wächst um Faktor 64 auf $3.0\cdot10^{-3}$ bis $1.8\cdot10^{-2}$ für $z_{01}\in[50, 300]$ mm, also $1/\mathrm{Fr}\in[55, 333] \le 512$. `src.data.generate_data` prüft $\mathrm{Fr}_\min \ge 1/(\texttt{detector\_size}/\texttt{downsample\_factor}\cdot\texttt{padding\_factor})$ vor der Erzeugung (`propagator_sampling_check`, Warnung bei Verletzung) und protokolliert das Ergebnis in `meta.json` (`propagator_sampling`). `configs/data_p05.yaml` (2048 px, Padding 4, Gitter 8192 ⇒ $\mathrm{Fr} \ge 1.2\cdot10^{-4}$) beginnt deshalb bei $z_{01} = 150$ mm ($\mathrm{Fr} \ge 1.4\cdot10^{-4}$). Beachte: Durch das Binning liegt `data_small` in einem anderen $\mathrm{Fr}$-Regime als P05 ($N_\ast = 1/\sqrt{\mathrm{Fr}} \approx 7$–$18$ px statt $46$–$84$ px bei `data_p05`); es eignet sich für Funktions- und Methodentests, Aussagen zur P05-Geometrie brauchen die P05-Konfiguration.

---

## 4. Kegelstrahlgeometrie und Fresnel-Skalierungstheorem

### 4.1 Größen

| Symbol | Bedeutung | HoloWizard | Typischer Wert P05 |
|---|---|---|---|
| $z_{01}$ | Fokus–Objekt-Abstand | `Measurement.z01` (mm) | 20–500 mm |
| $z_{02}$ | Fokus–Detektor-Abstand | `BeamSetup.z02` (mm) | 19 650–20 000 mm |
| $z_{12} = z_{02}-z_{01}$ | Objekt–Detektor-Abstand | intern | ≈ 19.5 m |
| $M = z_{02}/z_{01}$ | geometrische Vergrößerung | `ConeBeam` | 40–1000 |
| $\Delta x$ | physikalische Pixelgröße | `BeamSetup.px_size` (mm) | 0.0065 mm |
| $\Delta x_\mathrm{eff} = \Delta x/M$ | effektive Pixelgröße in der Objektebene | `dx_eff` | 6.5–160 nm |
| $z_\mathrm{eff} = z_{12}/M$ | effektive Propagationsdistanz | `z_eff` | 20–490 mm |
| $\mathrm{Fr}$ | Pixel-Fresnel-Zahl | `calc_Fr`, `ConeBeam.get_fr` | $2\cdot10^{-5}$–$5\cdot10^{-4}$ |

### 4.2 Fresnel-Skalierungstheorem und Herleitung der HoloWizard-Formel

Das Fresnel-Skalierungstheorem (Paganin 2006; in [2] Gl. (5)–(7)) besagt: Die Kegelstrahl-Propagation von der Objektebene zur Detektorebene ist äquivalent zu einer *Parallelstrahl*-Propagation über die effektive Distanz $z_\mathrm{eff} = z_{12}/M$, gefolgt von einer geometrischen Vergrößerung um $M$. In [2] werden zwei Faktoren eingeführt, $M_1 = z_{02}/z_{01}$ (Skalierung der Pixelgröße) und $M_2 = (z_{01}+z)/z_{01}$ (Skalierung der Propagationsdistanz $z$), mit
$$\mathrm{Fr}(z) = \frac{(\Delta x/M_1)^2}{\lambda\,(z/M_2)}.$$
Für die Propagation bis zum Detektor ist $z = z_{12}$, also $M_2 = z_{02}/z_{01} = M_1 = M$, und es folgt die in HoloWizard implementierte Formel:
$$\boxed{\ \mathrm{Fr} = \frac{\Delta x_\mathrm{eff}^2}{\lambda z_\mathrm{eff}} = \frac{(\Delta x/M)^2}{\lambda\,z_{12}/M} = \frac{\Delta x^2}{\lambda\,(z_{02}-z_{01})\,M} = \frac{\Delta x^2\,z_{01}}{\lambda\,z_{02}\,(z_{02}-z_{01})}\ }$$
Genau so steht es in `forge/utils/utilities.py::calc_Fr` (`M = z02/z01; Fr = px**2/(lam*(z02-z01)*M)`, nach Umrechnung aller Längen in nm) und in `core/models/cone_beam.py::ConeBeam.get_effective_geometry` (`dx_eff = px/M; z_eff = z12/M; fr_eff = dx_eff**2/lam/z_eff`). Beide Implementierungen liefern identische Werte (von mir für $z_{01} \in \{20,50,100,250,500,1000\}$ mm geprüft).

### 4.3 Zahlenbeispiel

$E = 11$ keV, $z_{01} = 250$ mm, $z_{02} = 20\,000$ mm, $\Delta x = 6.5$ µm:
- $\lambda = 1.2398/11 = 0.112709$ nm $= 1.12709\cdot10^{-7}$ mm
- $M = 20\,000/250 = 80$
- $\Delta x_\mathrm{eff} = 6.5\ \mathrm{µm}/80 = 81.25$ nm
- $z_{12} = 19\,750$ mm, $z_\mathrm{eff} = 19\,750/80 = 246.875$ mm
- $\mathrm{Fr} = (8.125\cdot10^{-5}\ \mathrm{mm})^2 / (1.12709\cdot10^{-7}\ \mathrm{mm}\cdot 246.875\ \mathrm{mm}) = 6.6016\cdot10^{-9} / 2.7825\cdot10^{-5} = 2.3725\cdot10^{-4}$

`calc_Fr(11.0, 250.0, 20000.0, 0.0065)` gibt `2.37252e-4`, `ConeBeam.get_fr` ebenfalls; `ConeBeam.get_z01(setup, 2.37252e-4)` liefert $249.9995$ mm zurück (Rundung auf nm im Code).

### 4.4 Tabelle $\mathrm{Fr}(z_{01})$ für die Standardgeometrie

| $z_{01}$ / mm | $M$ | $\Delta x_\mathrm{eff}$ / nm | $z_\mathrm{eff}$ / mm | $\mathrm{Fr}$ | $1/\mathrm{Fr}$ (Mindestgitter) |
|---|---|---|---|---|---|
| 20 | 1000 | 6.50 | 19.980 | $1.8762\cdot10^{-5}$ | 53 300 |
| 50 | 400 | 16.25 | 49.875 | $4.6975\cdot10^{-5}$ | 21 288 |
| 100 | 200 | 32.50 | 99.500 | $9.4186\cdot10^{-5}$ | 10 617 |
| 250 | 80 | 81.25 | 246.875 | $2.3725\cdot10^{-4}$ | 4 215 |
| 500 | 40 | 162.50 | 487.500 | $4.8059\cdot10^{-4}$ | 2 081 |

Über den Bereich 50–500 mm ändert sich $\mathrm{Fr}$ also um eine Größenordnung (nahezu linear in $z_{01}$, weil $z_{01} \ll z_{02}$); $\log\mathrm{Fr}$ ist deshalb ein gut skaliertes Regressionsziel.

Zum Abgleich mit [2], Tab. 2 (gleiche Formel, $\Delta x = 6.5$ µm): Magnesiumdraht 11 keV, 470.8 mm, 19.661 m → nachgerechnet $4.678\cdot10^{-4}$ (Paper: $4.678\cdot10^{-4}$); Draht in Flusszelle 329.3 mm, 19.914 m → $3.165\cdot10^{-4}$ (Paper identisch); Spinnenhaar 79.4 mm, 19.661 m → $7.73\cdot10^{-5}$ (Paper $7.79\cdot10^{-5}$, Abweichung 0.8 %); Simulation 100 mm, 20 m → $9.42\cdot10^{-5}$ (Paper $9.33\cdot10^{-5}$, 0.9 %). Die Formel ist also dieselbe; die Abweichungen unter 1 % erklären sich plausibel durch gerundete Tabellenwerte, sind aber nicht abschließend geklärt (Frage an die Betreuer).

### 4.5 Warum $z_{01}$ die unsichere Größe ist

- Logarithmische Sensitivität aus der Formel: $\dfrac{\partial\ln\mathrm{Fr}}{\partial z_{01}} = \dfrac{1}{z_{01}} + \dfrac{1}{z_{02}-z_{01}}$ und $\dfrac{\partial\ln\mathrm{Fr}}{\partial z_{02}} = -\dfrac{1}{z_{02}} - \dfrac{1}{z_{02}-z_{01}}$.
  Bei $z_{01} = 250$ mm: $0.405\,\%$ pro mm in $z_{01}$, aber nur $0.010\,\%$ pro mm in $z_{02}$. Bei $z_{01} = 79.4$ mm (Spinnenhaar in [2]) sogar $1.26\,\%$ pro mm. Ein Fehler von $\pm5$ mm in $z_{01}$ verändert $\mathrm{Fr}$ um 2–6 %, derselbe Fehler in $z_{02}$ um 0.05 % – genau die Aussage von Fig. 3 in [2].
- Messtechnisch: $z_{02}$ (ca. 20 m) ist über die Beamline-Mechanik auf Millimeter bekannt und ändert sich selten; $z_{01}$ ist der Abstand zwischen dem unsichtbaren FZP-Fokus und der Probe, wenige Zentimeter bis Dezimeter, wird mit Motorpositionen und Referenzmessungen auf etwa $\pm5$ mm geschätzt ([2]: $\Delta_M = \pm 5$ mm) und ändert sich mit jeder Probe, jeder Probenwechsel-Position und bei In-situ-Zellen.
- Energie und Pixelgröße sind auf $<0.1\,\%$ bekannt; das Vorzeichen bzw. die Abhängigkeit der Fresnel-Zahl von ihnen ist über $\lambda$ bzw. $\Delta x^2$ klar, spielt für die Fokussierung aber keine Rolle, weil [2] ausdrücklich nur $z_{01}$ optimiert (die Methode wäre auf jeden Fr-Parameter übertragbar).

### 4.6 Inverse Beziehung

Aus $\mathrm{Fr}$ (und bekanntem $z_{02}$, $\lambda$, $\Delta x$) folgt $z_{01}$ eindeutig (`ConeBeam.get_z01`):
$$z_{01} = \frac{\mathrm{Fr}\,\lambda\,z_{02}^2}{\Delta x^2 + \mathrm{Fr}\,\lambda\,z_{02}}.$$
Daraus die Fehlerfortpflanzung für ein Netz, das $\mathrm{Fr}$ schätzt: $\delta z_{01} \approx z_{01}\,(1 - z_{01}/z_{02})\,\delta\mathrm{Fr}/\mathrm{Fr}$. Ein relativer Fr-Fehler von 0.4 % entspricht bei 250 mm etwa 1 mm; 0.04 % entsprechen 0.1 mm (dem Abbruchkriterium in [2]).

---

## 5. Phasenproblem und Phase Retrieval

### 5.1 Das Phasenproblem

Gemessen wird $I_\mathrm{det} = |\mathcal D_\mathrm{Fr}(e^{i\tilde O}P)|^2$. Gesucht ist $\tilde O$ (zwei reelle Felder $\varphi,\mu$) aus einem reellen Feld $I_\mathrm{det}$: Das Problem ist unterbestimmt und nichtlinear (Betragsquadrat), außerdem schlecht gestellt (die CTF hat Nullstellen, Abschnitt 5.2). Eindeutigkeit und Stabilität folgen erst aus A-priori-Wissen: Nichtnegativität/Vorzeichen ($\Omega_P$), Support, Glattheit, Einmaterialannahme ($\delta/\beta$ konstant), Mehrfachabstände (Holotomographie, Cloetens et al. 1999) oder Regularisierung (Maretzke & Hohage 2020).

### 5.2 Kontrastübertragungsfunktion (CTF) für schwache Objekte

Für $|\varphi|, \mu \ll 1$ und ebene Beleuchtung linearisiert man $e^{i\tilde O}\approx 1 + i\varphi - \mu$ und erhält für den Kontrast $\mathcal F[I/I_0 - 1]$ (Guigay 1977; Cloetens et al. 1999; Zabler et al. 2005):
$$\mathcal F\!\left[\frac{I}{I_0}-1\right](\xi,\eta) = 2\sin\!\Big(\frac{\pi(\xi^2+\eta^2)}{\mathrm{Fr}}\Big)\,\mathcal F[\varphi] - 2\cos\!\Big(\frac{\pi(\xi^2+\eta^2)}{\mathrm{Fr}}\Big)\,\mathcal F[\mu]$$
(Vorzeichen je nach Konvention). Phasenkontrast verschwindet bei $\xi^2+\eta^2 = n\,\mathrm{Fr}$, Absorptionskontrast bei $(n+\tfrac12)\mathrm{Fr}$. Inversion durch Tikhonov-regularisierte Division (Ein-Abstands-CTF mit $\delta/\beta$-Annahme oder Mehrfachabstände). In HoloTomoToolbox (Lohse et al. 2020) als `phaserec_ctf` implementiert. Für die in [2] untersuchten Objekte mit Phasenschüben bis $>6\pi$ ist die Linearisierung ungültig.

### 5.3 TIE und Paganin [3]

Die Transport-of-Intensity-Gleichung (Teague 1983) verknüpft die Intensitätsänderung entlang der Ausbreitung mit der Phase:
$$\frac{\partial I}{\partial z} = -\frac{\lambda}{2\pi}\,\nabla_\perp\cdot\big(I\,\nabla_\perp\varphi\big).$$
Paganin et al. [3] setzen ein homogenes Objekt voraus ($\delta/\beta$ konstant, Phase und Absorption proportional zur projizierten Dicke $T$) und lösen die TIE für einen einzigen kurzen Abstand geschlossen:
$$T(x,y) = -\frac{1}{\mu_\mathrm{lin}}\ln\!\Big(\mathcal F^{-1}\Big[\frac{\mathcal F[I/I_0]}{1 + \pi\lambda z\,(\delta/\beta)\,(f_x^2+f_y^2)}\Big]\Big),\qquad \text{in Pixeleinheiten: } 1 + \frac{\pi(\delta/\beta)(\xi^2+\eta^2)}{\mathrm{Fr}}.$$
Ein stabiler Tiefpass, ein FFT-Paar, extrem schnell, aber nur für $\mathrm{Fr}$ im „TIE-Regime“ (erste CTF-Nullstelle jenseits der relevanten Frequenzen) und homogene Objekte quantitativ. Für dich relevant: Der Paganin-Filter ist ein Beispiel dafür, wie direkt $\mathrm{Fr}$ in jede Rekonstruktion eingeht – ein falsches $\mathrm{Fr}$ verschiebt den Filter und damit die Schärfe.

### 5.4 Iterative Projektionsverfahren: ER und HIO [4]

Fienup [4] vergleicht Algorithmen, die abwechselnd Nebenbedingungen im Objektraum (Support, Nichtnegativität) und im Messraum (gemessener Betrag, hier $\sqrt{I_\mathrm{det}}$) erzwingen. *Error Reduction* (ER) ist die reine alternierende Projektion (monoton fallender Fehler, aber Stagnation), *Hybrid Input-Output* (HIO) nutzt eine Rückkopplung außerhalb des Supports ($\beta_\mathrm{HIO} \approx 0.5$–$1$) und konvergiert in der Praxis deutlich schneller; Fienup zeigt zudem, dass ER eng mit dem Steepest-Descent-Verfahren auf dem Messraumfehler verwandt ist und dass ER sowie Gerchberg–Saxton (zwei Intensitätsmessungen) monoton konvergieren. Für NFH wurden diese Ideen zu RAAR/Alternating-Projections-Varianten weiterentwickelt (Lohse et al. 2020). ASRM [1] ist in dieser Tradition ein *projiziertes Gradientenverfahren*: Gradientenschritt auf dem Datenfehler, dann Projektion auf $\Omega_P$.

### 5.5 Gradientenverfahren und Regularisierung

Modernes NFH-Phase-Retrieval minimiert ein Funktional
$$\mathcal L(\tilde O) = \tfrac12\big\|\,|\mathcal D_\mathrm{Fr}(e^{i\tilde O}P)| - \sqrt{I_\mathrm{det}}\,\big\|_2^2 + \mathcal R(\tilde O)$$
mit analytischem Gradienten (Wirtinger-Kalkül, in HoloWizard `core/reconstruction/gradients/analytical.py`: Fehler in der Detektorebene, Rückpropagation, Multiplikation mit $-i\,\overline{e^{i\tilde O}}\,\overline P$), Nesterov-Momentum, Regularisierung (L2 auf der Absorption, Glättungsfilter) und Nebenbedingungen (Huhn et al. 2022 für Tikhonov-Varianten; Maretzke & Hohage 2020 zur Theorie). Der Datenfehler wird in der *Amplitude* ($\sqrt I$) gemessen, nicht in der Intensität – so auch in [2] Gl. (14), (20).

### 5.6 ASRM [1]

ASRM (Artifact-Suppressing Reconstruction Method, Dora et al. 2024) erweitert den PGD-Algorithmus „refAP“ (direkte Rekonstruktion des projizierten Brechungsindex ohne $2\pi$-Phasenmehrdeutigkeit) um:
- Verzicht auf einen räumlichen Support; stattdessen Nebenbedingungsmenge $\Omega_P = \{\operatorname{Re}\tilde O \le 0,\ \operatorname{Im}\tilde O \ge -\log A_0\}$ mit Intensitätsoffset $A_0$ (fängt Flatfield-Restfehler ab);
- Spiegel-Padding (`Padding.MIRROR_ALL`) und Fensterung der Daten zum Rand (`window_type="blackman"`), damit das Objekt den Bildrand überschreiten darf;
- getrennte Regularisierung von Phase und Absorption (L2 auf $\operatorname{Im}\tilde O$), Stabilisierung hoher Frequenzen (Gauß-Filter auf dem Update, FWHM getrennt für Real- und Imaginärteil);
- Nesterov-Beschleunigung mit hochfrequenzunterdrücktem Momentum;
- Multigrid: Start auf 16-fach heruntergetastetem Gitter, dann 4-fach (die drei Stufen 700/300/500 Iterationen in [2] Tab. 3 entsprechen exakt den `Options`-Stufen in `core/scripts/examples/find_focus/magnesium_wire.py`).

Im Kontext (`core/reconstruction/single_projection/context.py`) wird $\mathrm{Fr}$ pro Stufe mit der heruntergetasteten Pixelgröße neu berechnet ($\mathrm{Fr}\propto\Delta x^2$, bei 16× also Faktor 256). ASRM ist das Arbeitspferd an P05 und zugleich der „innere“ Löser des Autofokus in [2].

### 5.7 Folgen einer falschen Fresnel-Zahl

Ein falsches $\mathrm{Fr}$ im Vorwärtsmodell bedeutet: Die Rückpropagation landet nicht in der Objektebene, sondern in einer defokussierten Ebene $\hat z_{01} = z_{01} + \Delta_{01}$. Die Rekonstruktion zeigt dann (Fig. 2 in [2]) Unschärfe, Fresnel-Säume an Kanten („fringe artifacts“), Verlust an Auflösung, Verletzungen der Nebenbedingungen (positive Phasen, negative Absorption) und einen höheren Datenfehler. Mit $0.4\,\%$ Fr-Änderung pro mm (bei 250 mm) sind $\pm5$ mm bereits deutlich sichtbar; die Autoren von [2] setzen die Zielgenauigkeit auf $0.1$ mm.

---

## 6. Das Autofokus-Problem

### 6.1 Formulierung ([2] Abschnitt 3.2)

Gegeben ein Hologramm $I_\mathrm{det}$, eine Schätzung $z_{01}^\mathrm{est}$ und eine Unsicherheit $\Delta_M$ (P05: $\pm5$ mm). Gesucht:
$$z_{01}^\ast = \operatorname*{argmin}_{\hat z_{01}\in\Omega_z} f(\hat z_{01}),\qquad \Omega_z = [z_{01}^\mathrm{est}-\Delta_M,\ z_{01}^\mathrm{est}+\Delta_M],$$
wobei $f$ ein Fokuskriterium ist, das (i) sein Minimum bei $z_{01}$ hat und (ii) mit dem Abstand vom Optimum wächst. [2] zerlegt $f$ in ein Kriterium $G$ und die Rekonstruktion $\tilde O(\hat z_{01})$ und erhält das geschachtelte Problem
$$z_{01}^\ast = \operatorname*{argmin}_{\hat z_{01}\in\Omega_z} G\Big(\operatorname*{argmin}_{\tilde O\in\Omega_P} \mathcal L(\tilde O,\hat z_{01})\Big),\qquad \mathcal L(\tilde O,\hat z_{01}) = \tfrac12\big\|\mathcal D^{z_{02}-\hat z_{01}}_\mathrm{Fr}(\tilde O) - \sqrt{S_\downarrow I_\mathrm{det}}\big\|_2^2 + \beta\,\|\operatorname{Im}\tilde O\|_2^2 .$$

### 6.2 Klassische Fokuskriterien

Alle folgenden Kriterien werten nur das rekonstruierte Bild aus (in HoloWizard als `core/find_focus/focus_loss_metrics.py` implementiert: `get_var`, `get_spec`, `get_gra`, `get_lap`, `get_tog`, `get_gog`):

| Kriterium | Idee | Befund in [2] (Simulation / Messdaten) |
|---|---|---|
| VAR | Bildvarianz (Kontrastmaximum) | glatt, aber ohne Peak (kubisch); nur Rauschrobustheit erfüllt |
| SPEC | gewichtete Spektralanalyse | plötzliche Peaks, verrauscht, Peakrichtung wechselt je Objekt; erfüllt keine Anforderung |
| GRA | Statistik des Gradienten 1. Ordnung | v-förmig in Simulation; auf Messdaten Nebenpeaks |
| LAP | Laplace (Gradient 2. Ordnung) | scharfer, abrupt auftretender Peak, stark rauschempfindlich |
| ToG | Tamura-Koeffizient des Gradienten (Sparsity) | v-förmig in Simulation; Nebenpeaks auf Messdaten |
| GoG | Gini-Index des Gradienten (Sparsity) | wie ToG |

Anforderungen aus [2] Abschnitt 3.7 an ein Kriterium, damit Nelder–Mead funktioniert: (i) deutlicher Peak, (ii) konsistente Peakrichtung innerhalb des Kriteriums, (iii) v- oder u-förmiger Verlauf im gesamten Unsicherheitsbereich, (iv) möglichst glatt.

### 6.3 Modellbasierter Autofokus: das MFE-Kriterium [2]

Kernidee: Statt Bildstatistik wird die *Konsistenz des Vorwärtsmodells* gemessen. Eine defokussierte Welle $\hat\psi = \mathcal D^{\Delta_{01}}(\psi_\mathrm{exit})$ enthält Fresnel-Säume mit Werten außerhalb von $\Omega_P$ (positive Phase, Amplitude $>1$). Die Projektion $\hat\psi_P = P_{\Omega_P}\hat\psi$ löscht diese Werte unwiderruflich; propagiert man $\hat\psi_P$ zum Detektor, passt das Ergebnis nicht mehr zum gemessenen Hologramm. Der *Model Fit Error* ist dieser Restfehler ([2] Gl. (14)/(15)):
$$\mathrm{MFE}(\hat z_{01}) = \Big\|\,\big|\mathcal D^{\hat z_{12}}_\mathrm{Fr}(\hat\psi_P)\big| - \sqrt{I_\mathrm{det}}\,\Big\|_2^2 .$$
In der Praxis ist $\hat\psi_P = \exp(i\tilde O^\ast(\hat z_{01}))$ das Ergebnis der ASRM-Rekonstruktion mit der (falschen) Distanz, und der MFE ist schlicht der *Datenterm der Rekonstruktion nach der letzten Iteration*. Eigenschaften laut [2]: v-förmig und glatt in Simulation (noise-free) und auf allen fünf Messdatensätzen; dort, wo die klassischen Kriterien Nebenpeaks zeigen, hat der MFE nur „minor saddle points“. Weil der MFE die Nebenbedingungen des Rekonstruktionsalgorithmus nutzt, ist er algorithmusabhängig – das ist gewollt: Er misst, ob *diese* Rekonstruktion das Modell erklärt.

Optimierer: Nelder–Mead (Downhill-Simplex, gradientenfrei), Simplex initialisiert mit den beiden Intervallenden $z^\mathrm{est}_{01}\pm\Delta_M$ (z. B. 74.8 mm / 84.8 mm), Nebenbedingung $\Omega_z$ durch Projektion der Eckpunkte, Abbruch, wenn der Simplex kürzer als 100 µm ist. Jeder Probepunkt = eine Multigrid-ASRM-Rekonstruktion auf 4-fach heruntergetastetem Gitter.

Ergebnisse ([2] Tab. 1): Für alle fünf Objekte (Spinnenhaar, Zahn, Kaktusnadel, Mg-Draht, Mg-Draht in Flusszelle) konvergiert NM auf den visuell bestimmten Referenzwert bis auf $\le 0.1$ mm (79.4/79.4, 80.9/81.0, 284.2/284.2, 470.8/470.7, 329.3/329.3 mm) mit 9–13 Probepunkten. Laufzeit pro Rekonstruktion 6 s (Gitter 8192² → 2084² gepaddet/heruntergetastet, Tabellenangabe) bzw. 16 s (14 336² → 3584²), insgesamt also rund 1–4 Minuten pro Datensatz („a few minutes“), Hardware im Text nicht spezifiziert. Energien 11/17 keV, $z_{01}$ von 79 bis 471 mm, $z_{02}$ 19.65–19.91 m, Belichtungen 0.8–1.5 s.

Grenzen: (a) Minuten pro Fokussierung – für Online-Rekonstruktion, Tomographie-Serien mit Driften und In-situ-Studien ein Engpass; (b) Nelder–Mead ist lokal, braucht eine u/v-förmige Kurve im Intervall und eine Startschätzung innerhalb $\pm\Delta_M$; (c) das Ergebnis hängt von den ASRM-Hyperparametern (Tab. 3), vom Offset $A_0$ und vom Padding ab; (d) Evaluation auf fünf Messdatensätzen und drei Phantomen, keine Statistik über viele Proben, Simulation ohne Rauschen/Flatfield-Artefakte; (e) nur $z_{01}$ wird optimiert ($z_{02}$, $E$, $\Delta x$ als bekannt angenommen; Erweiterung laut Autoren möglich); (f) keine Unsicherheitsangabe; (g) die „Ground Truth“ ist eine visuelle Fokussierung.

### 6.4 Was im HoloWizard-Code konkret existiert

| Baustein | Ort (Paket `holowizard`) | Inhalt |
|---|---|---|
| Öffentliche API | `core.api.functions.find_focus.find_focus(measurements, beam_setup, options, data_dimensions, ...)` | lädt Daten, bildet `torch.sqrt(data)` (Intensität → Amplitude) und ruft die interne Suche auf; Varianten `find_focus_flatfieldcorrection`, `find_focus_z01_a0*` |
| 1D-Suche über $z_{01}$ | `core.find_focus.find_focus_z01.find_focus` | `scipy.optimize.minimize(get_loss_reconstruction, z01_guess, method="Nelder-Mead", bounds=[z01_bounds], options={"xatol": z01_tol, "fatol": 1e6, "initial_simplex": bounds})`; Zielfunktion setzt `measurements[0].z01 = z01`, ruft `reconstruct(...)` (Multistage) und gibt `loss_se_all[-1]` zurück |
| 2D-Suche $z_{01}, a_0$ | `core.find_focus.find_focus_z01_a0` | gemeinsame NM-Suche (adaptiv) über $z_{01}$ und Offset $a_0$ |
| Orthogonale Suche | `core.find_focus.find_focus_z01_a0_orthogonal_search` | erst $a_0$ (xatol 0.01), dann $z_{01}$ |
| Klassische Kriterien | `core.find_focus.focus_loss_metrics` | VAR, SPEC, GRA, LAP, ToG, GoG auf Rekonstruktionen |
| Fokusserie (Brute Force) | `core.scripts.examples.focus_series_singledim.magnesium_wire` + `core.utils.plot_loss_chart` | Gitter über $\pm10$ mm, SE-Verlust plus alle Kriterien |
| Parameter | `core.parameters.Measurement(z01, z01_confidence=5.0)` → `z01_bounds`; `Options(z01_tol=0.1)`; `Padding(a0=1.0)`; `BeamSetup(energy, px_size, z02)` | Einheiten mm/keV; `z01_tol` = 0.1 mm entspricht dem 100-µm-Kriterium in [2] |
| Geometrie | `core.models.cone_beam.ConeBeam.get_fr / get_z01` | Formeln aus Abschnitt 4 |
| Online-Pfad | `livereco.server.find_focus.FindFocus`, `pipe.tasks.find_focus.FindFocusTask` | kapseln `find_focus` für Livereco/Pipelines |
| Referenzkonfiguration | `core/scripts/examples/find_focus/magnesium_wire.py` | `z01_guess=470.51`, `z01_confidence=10.0`, 11 keV, 6.5 µm, $z_{02}=19\,661$ mm, Padding 4, `a0=1.1`, Stufen 700/300/500 Iterationen |

Der „Verlust“ `loss_se_all[-1]` ist die mittlere quadratische Amplitudenabweichung über das FOV (`analytical.py`: $\sum_\mathrm{FOV}|\,|\mathcal D(\psi)| - \sqrt I\,|^2 / N$) nach der letzten Stufe – also der MFE aus [2] bis auf Normierung.

### 6.5 Laufzeit und Online-Anforderungen

Livereco (HoloWizard-Server für Online-Rekonstruktion an der Beamline) soll Projektionen quasi in Echtzeit rekonstruieren. Dafür muss $z_{01}$ *vor* der ersten Rekonstruktion bekannt sein. Anforderungen an einen lernbasierten Autofokus, abgeleitet aus [2] und dem Livereco-Pfad:
- Latenz: deutlich unter einer Rekonstruktion (Ziel $\ll 1$ s pro Hologramm auf GPU; CPU-Inferenz im Sekundenbereich akzeptabel).
- Genauigkeit: $\le 0.1$–$0.5$ mm (sub-Prozent in $\mathrm{Fr}$), mindestens aber gut genug, um das Suchintervall von $\pm5$ mm auf $\pm1$ mm zu verengen (Hybridansatz).
- Robustheit: Objektklasse, Rauschen, Probe-Struktur, Flatfield-Fehler, Energie 11/17 keV, $z_{01}$ 20–500 mm.
- Eingabe: rohes oder flatfield-korrigiertes Hologramm, 2048² (oder heruntergetastet), ohne Rekonstruktion.
- Ausgabe: Punktschätzung plus Unsicherheit, damit entschieden werden kann, ob der Wert direkt verwendet oder noch mit [2] verfeinert wird.

---

## 7. Lernbasierter Autofokus

### 7.1 Regression

Ein CNN $f_\theta$ bildet das Hologramm auf einen Skalar ab. Zielgrößen:
- $\mathrm{Fr}$ direkt: physikalisch die natürliche Größe (Abschnitt 3.1), aber über $10^{-5}$–$10^{-3}$ schlecht skaliert.
- $\log\mathrm{Fr}$: linearisiert die Skala, Fehler werden relativ (ein konstanter Fehler in $\log\mathrm{Fr}$ = konstanter relativer Fehler in $\mathrm{Fr}$ ≈ konstanter relativer Fehler in $z_{01}$).
- $z_{01}$ in mm: direkt interpretierbar, aber nur sinnvoll, wenn $z_{02}$, $E$, $\Delta x$ im Trainings- und Testsetup gleich sind oder als Zusatzeingabe mitgegeben werden (sonst lernt das Netz Geometrie, die nicht im Bild steht).
Standardisierung (z-Score) des Ziels, L1/Huber-Verlust für Robustheit gegen Ausreißer, MAE/RMSE in mm als Metriken. Literatur: Ren et al. 2018 (DH, Regressions-CNN, MAE ≈ 0.05 mm gegenüber ≈ 1.2–1.6 mm für kNN/SVM/MLP), Pitkäaho et al. 2019 (DHM, AlexNet/VGG16, RMS 6.4 mm bei 20-mm-Toleranz, 0.25–0.7 s CPU), Jaferzadeh et al. 2019 (Einzelzellen), FocusNET 2023 (Std ≈ 54 µm, 600× schneller als Rekonstruktionsstapel).

### 7.2 Klassifikation

Diskretisierung von $z_{01}$ in Bins (z. B. 1 mm) und Kreuzentropie; Vorteil: natürliche Mehrgipfligkeit und Konfidenzen, Nachteil: Auflösung durch Binbreite begrenzt, ordinaler Charakter geht verloren (Abhilfe: Erwartungswert über Softmax, ordinale Verluste). In der DH-Literatur üblich (Pitkäaho 2017/2019 als Klassifikation über wenige Ebenen).

### 7.3 Hybrid: ML-Initialisierung + modellbasierte Verfeinerung

Das Netz liefert $\hat z_{01}$ und $\hat\sigma$; der Nelder–Mead-Autofokus aus [2] startet mit $\Omega_z = [\hat z_{01} - c\hat\sigma, \hat z_{01} + c\hat\sigma]$ statt $\pm5$ mm. Erwartung: weniger Probepunkte (NM mit Abbruch bei 100 µm braucht etwa $\log_2(2\Delta_M/0.1)$ Halbierungen: von ±5 mm ≈ 7, von ±1 mm ≈ 4–5 Schritte) und keine Gefahr von Nebenminima. Der Hybrid ist das risikoärmste Ergebnis der Thesis, weil er die Genauigkeit von [2] behält und nur die Laufzeit senkt. Umsetzung direkt über `Measurement(z01=ẑ01, z01_confidence=c·σ̂)`.

### 7.4 Simulationsbasierte Inferenz (SBI) / Neural Posterior Estimation

SBI [6] behandelt den Simulator (HoloForge) als implizite Likelihood $p(I\mid z_{01})$ und lernt mit Paaren $(z_{01}^{(i)}, I^{(i)})$, $z_{01}^{(i)}\sim p(z_{01})$, direkt eine Approximation der Posterior $q_\phi(z_{01}\mid I)$ (NPE: bedingter Dichteschätzer wie Normalizing Flow oder Mischdichte-Netz; Alternativen NLE, NRE). Für Bilder braucht man ein *Embedding-Netz* (CNN), das $I$ auf wenige Merkmale komprimiert; `sbi` erlaubt es, dieses Netz end-to-end mitzutrainieren. Die Inferenz ist *amortisiert*: Nach dem Training kostet die Posterior für ein neues Hologramm nur einen Vorwärtsdurchlauf (plus Sampling).

Wichtige Konzepte aus [6]: Prior-Wahl ($z_{01}\sim\mathcal U[20, 500]$ mm oder log-uniform), *Prior Predictive Checks* (sehen simulierte Hologramme realistisch aus?), *Posterior Predictive Checks*, Diagnostik der Kalibrierung: *Simulation-Based Calibration* (SBC, Rangstatistik muss uniform sein), *Expected Coverage* (HPD-Intervalle mit nominaler Abdeckung $1-\alpha$ sollen den wahren Wert in $1-\alpha$ der Fälle enthalten), TARP, lokale Tests (L-C2ST); Ensembles von NPE-Netzen gegen Überkonfidenz; *Modellfehlspezifikation* (Messdaten ≠ Simulator) als Hauptrisiko, mit Checks wie Vergleich der Embedding-Statistiken von Mess- und Simulationsdaten. Für deine Arbeit ist SBI der formale Rahmen, um die „Schätzung von $z_{01}$ inklusive Unsicherheit“ aus der Ausschreibung umzusetzen; der Hybrid (7.3) nutzt diese Unsicherheit direkt.

Einfachere Unsicherheitsmodelle als Baselines: heteroskedastische Regression (Netz gibt $\mu, \log\sigma^2$ aus, Gauß-NLL; Kendall & Gal 2017), MC-Dropout, Deep Ensembles (Lakshminarayanan et al. 2017) – jeweils mit denselben Kalibrierungsdiagnostiken prüfen.

### 7.5 Eingaberepräsentationen

| Repräsentation | Pro | Contra |
|---|---|---|
| Rohes Hologramm (flatfield-korrigiert, auf Mittelwert 1 normiert) | vollständige Information | Dynamik der Probe/Flatfield-Struktur dominiert; Netz muss Streifenskala selbst extrahieren |
| $\log I$ oder $\sqrt I$ (Amplitude wie in HoloWizard Core) | komprimiert Dynamik, $\log$ macht multiplikative Probe additiv | Rauschen bei kleinen Intensitäten verstärkt |
| Leistungsspektrum $\lvert\mathcal F[I/\bar I - 1]\rvert^2$ (log-skaliert) | $\mathrm{Fr}$ steht explizit in den Ringradien $\xi_n^2 = n\mathrm{Fr}$ (Abschnitt 3.4); translationsinvariant; objektunabhängiger | verliert Phasen-/Ortsinformation; hohe Dynamik; Ringe nur bis $\xi < N\mathrm{Fr}/2$ aufgelöst |
| Radiales Mittel des Leistungsspektrums (1D-Profil über $\xi^2$) | sehr kompakt, Ringe werden periodisch in $\xi^2$ mit Periode $\mathrm{Fr}$ → fast ein Peak-Finding-Problem; CPU-tauglich | setzt isotrope Statistik voraus (Objekte mit Vorzugsrichtung, anisotrope Probe) |
| Mehrkanal (Bild + Spektrum) | kombiniert Stärken | mehr Rechenaufwand |
| Multi-Crop / Patches | 2048² zu groß für GPU-Speicher → Patches von 256–512 px; Aggregation per Mittel/Median | weniger Ringe pro Patch (3.4) |

### 7.6 Invarianzen und Augmentierung

- $\mathrm{Fr}$ ist invariant gegen Translation, Spiegelung, 90°-Rotation des Hologramms → erlaubte Augmentierungen (plus Crops).
- $\mathrm{Fr}$ ist *nicht* invariant gegen Skalierung: Resize um Faktor $s$ ändert die scheinbare Fresnel-Zahl um $s^2$ (Fresnel-Skalensymmetrie). Also kein Resize, keine Zoom-Augmentierung, kein Interpolieren auf andere Gitter; Downsampling nur mit korrekt umgerechnetem Label (HoloWizard macht genau das: $\mathrm{Fr}$ mit `px·downsample_factor`).
- Invarianz gegen Objekt, Probe, Rauschen, $A_0$-Offset muss *gelernt* werden → Domänenrandomisierung in der Simulation (Tobin et al. 2017): zufällige Formen, Materialien, Dicken, Probe-Polynome, Rauschpegel, Flatfields.
- Objektgrößen-Dilemma: Große, glatte Objekte liefern wenige, kontrastarme Ringe; sehr kleine Objekte wenig Signal. Teste Robustheit explizit über Objektgröße und Material.

### 7.7 Sim-to-real und die Rolle von HoloForge

HoloForge [5] (Paket `holowizard.forge`) ist der Simulator: `PhantomGenerator` (Zufallsformen Rectangle/Polygon/Ellipse/Ball/Cylinder/CylinderRoundTip, Materialien über `xraylib`, Dicken in µm, Glättung), `ProbeGenerator` (polynomiale Beleuchtung, optional Rauschen), `NFHSimulation` (Propagation mit `create_kernel(Fr)`), `HologramGenerator` (Rauschen), `HDF5Labeller` (Layout `images/hologram`, optional `images/phantoms|probe|flatfield|gt_hologram`, `metadata/setup/{Fr, z01, z02, energy, detector_px_size, ...}`), CLI `forge/scripts/generate_data.py config output_dir num_samples --override --seed`. Standardkonfiguration `forge/configs/default.json`: 2048 px Detektor, Padding, Mg-Phantome, Gaußrauschen.

Sim-to-real-Lücke: reale Hologramme haben strukturierte Probes (FZP-Beugung, Speckle), Flatfield-Drift, Szintillator-Unschärfe, Detektor-Nichtlinearitäten, Vibrationen, teilkohärente Beleuchtung und Objekte außerhalb der Formenbibliothek. Gegenmaßnahmen: gemessene Flatfields in Forge einspeisen (`flatfield_dataset`, TIFF), Rauschpegel randomisieren, spektrale Eingaben (weniger Probe-abhängig), Fine-Tuning auf wenigen Messhologrammen mit Referenz-$z_{01}$ aus [2], Diagnostik der Fehlspezifikation [6]. Die Simulation ist in Pixeleinheiten exakt dieselbe Mathematik wie der ASRM-Vorwärtsoperator – mit einem Unterschied in der Ausgabe: `NFHSimulation._make_hologram` gibt `torch.abs(psi_det)` zurück (Amplitude), während HoloWizard Core gemessene Intensitäten über `torch.sqrt` in Amplituden umrechnet. Für Training auf Forge-Daten und Test auf Messdaten musst du daher $\sqrt{I_\mathrm{mess}}$ verwenden (oder Forge-Ausgaben quadrieren), sonst passen die Statistiken nicht.

---

## 8. Paper [2] vs. HoloWizard-Code: Konventionen und Abweichungen

| Punkt | Paper [2] / [1] | HoloWizard-Code | Konsequenz |
|---|---|---|---|
| Fresnel-Zahl | $\mathrm{Fr}(z)$ mit $M_1, M_2$ (Gl. 5–7), für $z = z_{12}$ identisch | `calc_Fr`, `ConeBeam.get_fr`: $\Delta x^2 z_{01}/(\lambda z_{02} z_{12})$ | identisch; Tabellenwerte in [2] bis <1 % reproduzierbar |
| Messgröße | $I_\mathrm{det}$, Vergleich mit $\sqrt{I_\mathrm{det}}$ | Core: `sqrt(data)`; Forge: `abs(psi)` als „hologram“ | Forge-Ausgabe ist Amplitude, Messdaten sind Intensität |
| Absorption | $\mu = k\int\beta\,dz$ | Forge: `2*beta*k*t` im Exponenten der Amplitude | Faktor 2 (zu klären) |
| Kriterium | MFE, Gl. (14)/(15) | `loss_se_all[-1]` (Amplituden-MSE über FOV) | gleiches Kriterium, Normierung per Pixel |
| Abbruch | Simplexlänge 100 µm | `Options.z01_tol = 0.1` mm (`xatol`), `fatol=1e6` (praktisch aus) | identisch |
| Simplex-Init | Intervallenden $z^\mathrm{est}\pm\Delta_M$ | `initial_simplex = bounds_array`, `bounds` aus `z01_confidence` | identisch (Standard 5 mm, Beispiel 10 mm) |
| Offset | $A_0$ in $\Omega_P$ | `Padding.a0` (Bild wird durch `a0` geteilt und mit `a0` gepaddet), zusätzlich 2D-Suche über `a0` | Code geht über das Paper hinaus |
| Multigrid-Stufen | 700/300/500 It., 16×/4×/4× | `Options`-Liste in den Beispielskripten | identisch |
| Flatfield | nicht Teil des Autofokus | `find_focus_flatfieldcorrection` (Komponentenmodell) | Code geht über das Paper hinaus |

---

## 9. Die Aufgaben der Ausschreibung als Forschungsfragen mit messbaren Kriterien

Die Ausschreibung nennt (1) Background Review ([3], [4], [5], [2]), (2) Data creation & Modeling (Trainingsdaten mit HoloForge oder Messdaten; Modelle/Strategien; Einbeziehung von Schätzungen für $z_{01}$ z. B. per SBI [6]) und (3) Implement & Test auf synthetischen und Messdaten. Daraus:

| Nr. | Forschungsfrage | Messbares Kriterium |
|---|---|---|
| F1 | Kann ein CNN $\mathrm{Fr}$/$z_{01}$ aus einem einzelnen simulierten Hologramm (P05-Geometrie, $z_{01}\in[50,500]$ mm) schätzen? | MAE$(z_{01}) \le 1$ mm (Minimalziel), $\le 0.5$ mm (Ziel), $\le 0.1$ mm (entspricht [2]); relativer Fr-Fehler median $\le 0.4\,\%$ bzw. $\le 0.2\,\%$; Anteil $|\Delta z_{01}| \le 1$ mm $\ge 95\,\%$ |
| F2 | Welche Eingaberepräsentation (roh, log, Amplitude, Leistungsspektrum, radiales Profil, Kombination) ist am genauesten und robustesten? | gleiche Metriken je Repräsentation bei identischem Budget; Robustheitskurven über Rauschpegel, Objektgröße, Material |
| F3 | Welche Zielparametrisierung ($\mathrm{Fr}$, $\log\mathrm{Fr}$, $z_{01}$) und welcher Verlust sind am besten? | MAE in mm über den gesamten $z_{01}$-Bereich; Fehler-vs-$z_{01}$-Kurve (keine systematische Drift an den Rändern) |
| F4 | Liefert SBI/NPE eine kalibrierte Posterior $q(z_{01}\mid I)$? | Expected Coverage innerhalb $\pm5$ Prozentpunkten der Nominalabdeckung bei 50/80/95 %; SBC-Ränge uniform (KS-Test $p>0.05$); Korrelation zwischen Posterior-Std und tatsächlichem Fehler $>0.5$; Posterior-Breite $\le 2$ mm (95 %) im Mittel |
| F5 | Beschleunigt die ML-Schätzung den modellbasierten Autofokus [2] ohne Genauigkeitsverlust (Hybrid)? | Zahl der NM-Rekonstruktionen von 9–13 auf $\le 5$; Endergebnis innerhalb $0.1$ mm des Ergebnisses mit $\pm5$-mm-Start; Gesamtlaufzeit mindestens halbiert |
| F6 | Wie groß ist die Sim-to-real-Lücke auf P05-Messdaten? | MAE auf Messhologrammen gegen Referenz-$z_{01}$ aus [2]/Fokusserie: $\le 1$ mm ohne Fine-Tuning als Ziel; mit Fine-Tuning $\le 0.5$ mm; Fehlspezifikations-Diagnostik dokumentiert |
| F7 | Reicht die Genauigkeit für die Rekonstruktion aus (Downstream)? | ASRM-Rekonstruktion mit $\hat z_{01}$: MFE $\le 1.05\times$ MFE am Referenzwert; PSNR/SSIM der Phase gegen Referenzrekonstruktion; visuelle Prüfung der Kantensäume |
| F8 | Laufzeit | Inferenz pro Hologramm $\le 50$ ms (GPU) bzw. $\le 2$ s (CPU) inklusive Vorverarbeitung, gegenüber Minuten in [2] |

---

## 10. Beitrag der Thesis und Nicht-Ziele

Beitrag:
1. Ein reproduzierbarer Daten- und Trainingsstack auf Basis von HoloForge für die Aufgabe „Hologramm → $\mathrm{Fr}$/$z_{01}$“ (randomisierte Geometrie, HDF5, Splits, Metadaten), bereits begonnen (Phase 0/1 im Repository: `NFHRandomDistSetup`, `HologramHDF5Dataset`, `AutofocusCNN`, `src/utils/physics.py`, `src/evaluate.py`, Konfigurationen `data_small.yaml` (256 px, CPU) / `data_p05.yaml` (2048 px, GPU)).
2. Eine systematische Studie zu Eingaberepräsentationen, Zielparametrisierungen und Architekturen für den lernbasierten Autofokus in NFH – in der Literatur bisher nur für sichtbare DH/DHM, nicht für Röntgen-NFH mit Kegelstrahlgeometrie.
3. Eine probabilistische Variante (SBI/NPE oder heteroskedastische Regression) mit geprüfter Kalibrierung und ein Hybridverfahren, das die Unsicherheit als Suchintervall an den modellbasierten Autofokus [2] übergibt.
4. Evaluation auf P05-Messdaten gegen den Stand der Technik [2] in Genauigkeit *und* Laufzeit, inklusive Downstream-Rekonstruktionsqualität.

Nicht-Ziele (bewusst außerhalb):
- Kein neuer Phase-Retrieval-Algorithmus; ASRM/HoloWizard bleibt der Rekonstruktionspfad.
- Keine Ende-zu-Ende-Rekonstruktion per Netz (Hologramm → Phase); nur der Skalar $\mathrm{Fr}$/$z_{01}$ (ggf. $A_0$ als Nebenprodukt).
- Keine Optimierung von $z_{02}$, Energie oder Pixelgröße (sie werden als bekannt angenommen, wie in [2]).
- Keine Änderung des HoloWizard-Pakets; Integration erfolgt über dessen öffentliche API (`Measurement.z01`, `z01_confidence`).
- Keine Tomographie-Rekonstruktion; Einzelprojektionen genügen.
- Keine Garantie der Übertragbarkeit auf andere Beamlines ohne neue Simulation (Geometrie/Energie sind Teil des Priors).

---

## 11. Glossar

| Begriff | Erklärung |
|---|---|
| $A_0$ / `a0` | Intensitätsoffset zur Korrektur imperfekter Flatfield-Normierung; untere Schranke $-\log A_0$ für $\operatorname{Im}\tilde O$ in ASRM |
| ASRM | Artifact-Suppressing Reconstruction Method [1]: projiziertes Gradientenverfahren auf dem projizierten Brechungsindex ohne Support, mit Spiegel-Padding, Regularisierung, Nesterov-Momentum, Multigrid |
| Amortisierte Inferenz | Einmaliges Training eines Netzes, das danach für jede neue Beobachtung die Posterior ohne erneute Optimierung liefert (NPE) |
| Austrittswelle $\psi_\mathrm{exit}$ | komplexe Welle direkt hinter dem Objekt, $\exp(i\tilde O)P$ |
| $\beta$ | Imaginärteil-Dekrement des Brechungsindex (Absorption) |
| CTF | Kontrastübertragungsfunktion; linearisiertes Vorwärtsmodell für schwache Objekte mit $\sin/\cos(\pi\lvert\xi\rvert^2/\mathrm{Fr})$ |
| $\delta$ | Realteil-Dekrement des Brechungsindex (Phasenschub) |
| $\delta/\beta$ | Materialkennzahl; Grundlage der Paganin-Methode |
| Deep Ensemble | mehrere unabhängig trainierte Netze; Streuung der Vorhersagen als epistemische Unsicherheit |
| Domänenrandomisierung | Zufällige Variation von Simulationsparametern, damit Netze auf reale Daten verallgemeinern |
| ER / HIO | Error Reduction / Hybrid Input-Output: klassische iterative Phase-Retrieval-Algorithmen [4] |
| Expected Coverage | Diagnose: Anteil der Testfälle, in denen das $(1-\alpha)$-Posteriorintervall den wahren Wert enthält, gegen $1-\alpha$ aufgetragen |
| Flatfield | Leeraufnahme $\lvert P\rvert^2$ ohne Objekt; dient der Normierung des Hologramms |
| Fokusserie | Brute-Force-Scan über $z_{01}$ mit je einer Rekonstruktion; Referenzverfahren und Visualisierung der Fokuskurve |
| Fresnel-Propagator $\mathcal D_\mathrm{Fr}$ | Freiraumausbreitung als Multiplikation mit $\exp(-i\pi\lvert\xi\rvert^2/\mathrm{Fr})$ im Fourierraum |
| Fresnel-Skalierungstheorem | Kegelstrahl = Parallelstrahl mit $\Delta x/M$ und $z_{12}/M$, anschließend Vergrößerung $M$ |
| Fresnel-Zahl (Pixel) $\mathrm{Fr}$ | $\Delta x_\mathrm{eff}^2/(\lambda z_\mathrm{eff})$; einziger Skalar, der Geometrie und Wellenlänge im Vorwärtsmodell bündelt |
| FZP | Fresnel-Zonenplatte, Fokussieroptik bei P05; ihr Fokus ist die virtuelle Punktquelle |
| GoG / ToG | Gini-Index bzw. Tamura-Koeffizient des Bildgradienten; Sparsity-basierte Fokuskriterien |
| GRA / LAP / VAR / SPEC | Fokuskriterien aus Gradient, Laplace, Varianz, Spektrum |
| HoloForge | Simulationspaket in HoloWizard (`holowizard.forge`) für synthetische Hologramme mit Labels |
| HoloWizard | Python-Framework (DESY/Hereon/TUHH) für Online-Rekonstruktion von NFH-Daten [5]; Pakete `core`, `forge`, `livereco`, `pipe` |
| Hologramm | gemessenes Fresnel-Beugungsbild (Intensität), in dem die Phase als Interferenzmuster kodiert ist |
| Hybridansatz | ML-Schätzung als Startwert/Intervall für den modellbasierten Autofokus |
| In-situ/operando | Messungen während eines Prozesses (z. B. Korrosion in Flusszelle); erfordern schnelle, automatische Fokussierung |
| Kohärenz | Fähigkeit der Welle zur Interferenz über Raum (transversal) und Zeit (longitudinal) |
| Livereco | HoloWizard-Server für Online-Rekonstruktion an der Beamline |
| $M$ | geometrische Vergrößerung $z_{02}/z_{01}$ |
| MFE | Model Fit Error [2]: Datenfehler $\lVert\,\lvert\mathcal D(\hat\psi_P)\rvert - \sqrt{I_\mathrm{det}}\rVert^2$ einer mit falschem $z_{01}$ rekonstruierten, auf $\Omega_P$ projizierten Welle |
| Multigrid | Rekonstruktion zunächst auf heruntergetasteten Gittern (16×, 4×), dann feiner |
| Nelder–Mead | gradientenfreies Downhill-Simplex-Verfahren; Optimierer in [2] und HoloWizard |
| NFH | Nahfeldholographie (near-field holography) |
| NPE / NLE / NRE | Neural Posterior / Likelihood / Ratio Estimation – die drei SBI-Grundfamilien [6] |
| $\Omega_P$ | Nebenbedingungsmenge: $\operatorname{Re}\tilde O\le 0$, $\operatorname{Im}\tilde O\ge-\log A_0$ |
| P05 | Imaging-Beamline an PETRA III (DESY, Hamburg), betrieben von Helmholtz-Zentrum Hereon; Nano-Imaging-Endstation |
| Padding | Vergrößerung des Rechengitters (Spiegelung), um Wrap-around der Propagation zu vermeiden; Bedingung $N\ge1/\mathrm{Fr}$ |
| Paganin-Methode | TIE-basiertes Einzelabstands-Phase-Retrieval für homogene Objekte [3] |
| Phasenproblem | Verlust der Phase bei der Intensitätsmessung; Rekonstruktion braucht A-priori-Wissen |
| Posterior Predictive Check | Prüfung, ob aus Posterior-Samples simulierte Daten der Beobachtung ähneln |
| Probe $P$ | komplexe Beleuchtungswelle in der Objektebene |
| Projektionsnäherung | Objekt als dünne Phasen-/Amplitudenmaske; Beugung im Objekt vernachlässigt |
| SBC | Simulation-Based Calibration: Rang des wahren Parameters unter Posterior-Samples muss uniform verteilt sein |
| SBI | Simulation-Based Inference: Bayessche Inferenz mit Simulator statt expliziter Likelihood [6] |
| Sim-to-real | Übertragung eines auf Simulationen trainierten Modells auf Messdaten |
| TIE | Transport-of-Intensity-Gleichung |
| $z_{01}, z_{02}, z_{12}$ | Fokus–Objekt, Fokus–Detektor, Objekt–Detektor-Abstand |
| $z_\mathrm{eff}$, $\Delta x_\mathrm{eff}$ | effektive Parallelstrahl-Distanz $z_{12}/M$ und effektive Pixelgröße $\Delta x/M$ |
