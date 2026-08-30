"""Build the German PDF documentation for the autofocus thesis project."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "Projektdokumentation_Learning_based_Autofocus.pdf"

NAVY = colors.HexColor("#15324A")
BLUE = colors.HexColor("#1F6E8C")
CYAN = colors.HexColor("#D9EEF3")
PALE = colors.HexColor("#F3F7F9")
GOLD = colors.HexColor("#D39B3B")
INK = colors.HexColor("#25313A")
MUTED = colors.HexColor("#60717D")
RED = colors.HexColor("#9F3A38")
GREEN = colors.HexColor("#287A55")


def register_fonts() -> None:
    fonts = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("DocSans", str(fonts / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("DocSans-Bold", str(fonts / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("DocMono", str(fonts / "consola.ttf")))
    pdfmetrics.registerFontFamily(
        "DocSans", normal="DocSans", bold="DocSans-Bold"
    )


class NumberedCanvasMixin:
    pass


class PipelineDiagram(Flowable):
    """Compact horizontal flow diagram drawn as vector graphics."""

    def __init__(self, labels: list[str], width: float = 168 * mm):
        super().__init__()
        self.labels = labels
        self.width = width
        self.height = 28 * mm

    def draw(self) -> None:
        canvas = self.canv
        count = len(self.labels)
        gap = 5 * mm
        box_width = (self.width - gap * (count - 1)) / count
        box_height = 16 * mm
        y = 6 * mm
        for index, label in enumerate(self.labels):
            x = index * (box_width + gap)
            canvas.setFillColor(CYAN if index % 2 == 0 else PALE)
            canvas.setStrokeColor(BLUE)
            canvas.roundRect(x, y, box_width, box_height, 2 * mm, fill=1)
            canvas.setFillColor(NAVY)
            canvas.setFont("DocSans-Bold", 7.5)
            lines = label.split("\n")
            start_y = y + box_height / 2 + (len(lines) - 1) * 4
            for line_index, line in enumerate(lines):
                canvas.drawCentredString(
                    x + box_width / 2, start_y - line_index * 9, line
                )
            if index < count - 1:
                arrow_x = x + box_width
                center_y = y + box_height / 2
                canvas.setStrokeColor(GOLD)
                canvas.setFillColor(GOLD)
                canvas.setLineWidth(1.5)
                canvas.line(arrow_x + 1 * mm, center_y, arrow_x + gap - 1 * mm, center_y)
                canvas.line(arrow_x + gap - 2.5 * mm, center_y + 1.5 * mm, arrow_x + gap - 1 * mm, center_y)
                canvas.line(arrow_x + gap - 2.5 * mm, center_y - 1.5 * mm, arrow_x + gap - 1 * mm, center_y)


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            fontName="DocSans-Bold",
            fontSize=27,
            leading=32,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="DocSans",
            fontSize=13,
            leading=18,
            textColor=MUTED,
            spaceAfter=6 * mm,
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName="DocSans-Bold",
            fontSize=19,
            leading=23,
            textColor=NAVY,
            spaceBefore=2 * mm,
            spaceAfter=5 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName="DocSans-Bold",
            fontSize=13,
            leading=17,
            textColor=BLUE,
            spaceBefore=5 * mm,
            spaceAfter=2.5 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="DocSans",
            fontSize=9.5,
            leading=14,
            textColor=INK,
            spaceAfter=2.6 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName="DocSans",
            fontSize=8,
            leading=11,
            textColor=MUTED,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName="DocSans",
            fontSize=9.2,
            leading=13.5,
            leftIndent=5 * mm,
            firstLineIndent=-3.5 * mm,
            bulletIndent=0,
            textColor=INK,
            spaceAfter=1.6 * mm,
        ),
        "code": ParagraphStyle(
            "Code",
            fontName="DocMono",
            fontSize=7.4,
            leading=10.5,
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            borderColor=colors.HexColor("#C8D6DD"),
            borderWidth=0.6,
            borderPadding=3 * mm,
            backColor=PALE,
            textColor=colors.HexColor("#203540"),
            spaceBefore=2 * mm,
            spaceAfter=3.5 * mm,
        ),
        "callout": ParagraphStyle(
            "Callout",
            fontName="DocSans",
            fontSize=9.2,
            leading=13.5,
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            borderColor=GOLD,
            borderWidth=0,
            borderLeft=3,
            borderPadding=3 * mm,
            backColor=colors.HexColor("#FFF8E8"),
            textColor=INK,
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "caption": ParagraphStyle(
            "Caption",
            fontName="DocSans",
            fontSize=7.8,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=3 * mm,
        ),
    }


def p(text: str, style, **kwargs) -> Paragraph:
    return Paragraph(text, style, **kwargs)


def bullet(text: str, st) -> Paragraph:
    return p(f"<bullet>&#8226;</bullet>{text}", st["bullet"])


def table(data, widths, header=True, font_size=8.2) -> Table:
    result = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "DocSans"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D6DD")),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "DocSans-Bold"),
        ]
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE))
    result.setStyle(TableStyle(commands))
    return result


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D8E2E7"))
    canvas.setLineWidth(0.5)
    canvas.line(21 * mm, height - 15 * mm, width - 21 * mm, height - 15 * mm)
    canvas.setFont("DocSans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(21 * mm, height - 11.5 * mm, "MASTERARBEIT - LEARNING-BASED AUTOFOCUS")
    canvas.drawRightString(width - 21 * mm, 11 * mm, f"Seite {doc.page}")
    canvas.setFillColor(BLUE)
    canvas.rect(21 * mm, 9.2 * mm, 16 * mm, 1.2 * mm, fill=1, stroke=0)
    canvas.restoreState()


def cover_page(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 55 * mm, width, 55 * mm, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, height - 58 * mm, width, 3 * mm, fill=1, stroke=0)
    canvas.setFillColor(BLUE)
    canvas.circle(width - 28 * mm, 25 * mm, 13 * mm, fill=1, stroke=0)
    canvas.setFillColor(CYAN)
    canvas.circle(width - 28 * mm, 25 * mm, 6 * mm, fill=1, stroke=0)
    canvas.restoreState()


def build_story(st):
    story = []
    story += [
        Spacer(1, 57 * mm),
        p("Projektdokumentation", st["title"]),
        p("Learning-based Autofocus for Holography", st["subtitle"]),
        HRFlowable(width="42%", thickness=2, color=GOLD, hAlign="LEFT"),
        Spacer(1, 8 * mm),
        p(
            "Technische Erklärung des vollständigen Python-Projekts zur CNN-basierten "
            "Schätzung des axialen Objekt-Detektor-Abstands aus simulierten "
            "Röntgenhologrammen.",
            st["body"],
        ),
        Spacer(1, 14 * mm),
        table(
            [
                ["Dokumenttyp", "Entwicklungs- und Code-Dokumentation"],
                ["Projektordner", "master_thesis_autofocus"],
                ["Zielgröße", "Abweichung von 1,00 m in Zentimetern"],
                ["Modell", "Convolutional Neural Network (Regression)"],
                ["Stand", "23. August 2026"],
            ],
            [42 * mm, 106 * mm],
            header=False,
            font_size=8.8,
        ),
        Spacer(1, 30 * mm),
        p(
            "Hinweis: Dieses Dokument beschreibt den aktuellen Prototypen. Es ist keine "
            "abschließende physikalische Validierung und kein Ersatz für die methodische "
            "Herleitung in der Masterarbeit.",
            st["small"],
        ),
        PageBreak(),
    ]

    story += [
        p("1. Kurzfassung", st["h1"]),
        p(
            "Das Projekt untersucht einen datengetriebenen Autofokus für die "
            "Röntgen-Nahfeldholographie. Statt den Fokusabstand für jedes Hologramm "
            "iterativ mit einem Optimierer wie Nelder-Mead zu suchen, soll ein CNN den "
            "Abstand in einem einzigen Vorwärtsdurchlauf schätzen. Der aktuelle Code ist "
            "ein funktionsfähiger Forschungsprototyp: Er erzeugt synthetische "
            "Phasenobjekte, simuliert deren Fresnel-Ausbreitung, trainiert einen "
            "Regressor und speichert Modell sowie Diagnosegrafiken.",
            st["body"],
        ),
        p("1.1 Kernidee", st["h2"]),
        PipelineDiagram(
            [
                "Phasenobjekt\nerzeugen",
                "Fresnel-\nPropagation",
                "Intensität\nnormalisieren",
                "CNN\ntrainieren",
                "Defokus\nschätzen",
            ]
        ),
        p(
            "Abbildung 1: Daten- und Modellpipeline des aktuellen Prototyps.",
            st["caption"],
        ),
        p("1.2 Ein- und Ausgaben", st["h2"]),
        table(
            [
                ["Element", "Form / Einheit", "Bedeutung"],
                ["CNN-Eingabe", "[B, 1, 128, 128]", "Batch aus normalisierten Graustufen-Hologrammen"],
                ["Trainingslabel", "[B], cm", "Signierte Abweichung relativ zu 1,00 m"],
                ["CNN-Ausgabe", "[B], cm", "Geschätzte axiale Abweichung"],
                ["Berichtsmetrik", "mm", "MAE nach Rückrechnung auf absolute Distanz"],
            ],
            [36 * mm, 42 * mm, 78 * mm],
        ),
        p("1.3 Was der Prototyp bereits leistet", st["h2"]),
        bullet("End-to-End-Ablauf von der Simulation bis zum gespeicherten Modell.", st),
        bullet("Konfigurierbare Datenmenge, Distanzspanne, Architektur und Lernparameter.", st),
        bullet("Regression statt Klassifikation: Der Fokuswert ist kontinuierlich.", st),
        bullet("Diagnose über Loss-Kurven, Streudiagramm und MAE in Millimetern.", st),
        PageBreak(),
    ]

    story += [
        p("2. Projektstruktur", st["h1"]),
        p(
            "Die Struktur trennt Konfiguration, Daten, Implementierung und Ergebnisse. "
            "Dadurch können Simulation, Modell und Auswertung später unabhängig erweitert "
            "werden.",
            st["body"],
        ),
        p(
            "master_thesis_autofocus/<br/>"
            "|-- configs/base.yaml<br/>"
            "|-- data/{raw, processed, external}/<br/>"
            "|-- notebooks/<br/>"
            "|-- src/data/generator.py<br/>"
            "|-- src/models/cnn.py<br/>"
            "|-- src/utils/{metrics.py, plotting.py}<br/>"
            "|-- src/train.py<br/>"
            "|-- reports/{figures, logs, trained_models}/<br/>"
            "|-- README.md, environment.yml, .gitignore",
            st["code"],
        ),
        table(
            [
                ["Pfad", "Verantwortung"],
                ["configs/base.yaml", "Zentrale Experiment- und Hyperparameter"],
                ["data/raw", "Unveränderte Mess- oder Simulationsdaten"],
                ["data/processed", "Vorverarbeitete, reproduzierbare Datensätze"],
                ["data/external", "Externe Referenzdaten oder Begleitmaterial"],
                ["src/data", "Physikalische Simulation und Dataset-Erzeugung"],
                ["src/models", "PyTorch-Netzarchitekturen"],
                ["src/utils", "Metriken und Visualisierungen"],
                ["reports/figures", "Loss- und Vorhersagegrafiken"],
                ["reports/trained_models", "Trainierte PyTorch-Gewichte"],
            ],
            [54 * mm, 102 * mm],
        ),
        p("2.1 Rolle der Paketdateien", st["h2"]),
        p(
            "Die vier <font name='DocMono'>__init__.py</font>-Dateien markieren "
            "Verzeichnisse als Python-Pakete. Dadurch sind Importe wie "
            "<font name='DocMono'>from src.models.cnn import AutofocusCNN</font> möglich. "
            "Sie enthalten aktuell keine öffentliche API und keine Initialisierungslogik.",
            st["body"],
        ),
        p("2.2 Versionskontrolle", st["h2"]),
        p(
            "Die <font name='DocMono'>.gitignore</font> schließt Python-Artefakte, lokale "
            "Umgebungen, große Daten, trainierte Gewichte und generierte Berichte aus. "
            "Das schützt das Repository vor unnötig großen oder maschinenspezifischen "
            "Dateien. Für wissenschaftliche Reproduzierbarkeit sollten wichtige Resultate "
            "trotzdem versioniert oder über ein Artefakt-Repository archiviert werden.",
            st["body"],
        ),
        PageBreak(),
    ]

    story += [
        p("3. Konfiguration und Umgebung", st["h1"]),
        p("3.1 Conda-Umgebung", st["h2"]),
        p(
            "Die Datei <font name='DocMono'>environment.yml</font> definiert die Umgebung "
            "<font name='DocMono'>holophd</font> mit Python 3.11, PyTorch, Torchvision, "
            "NumPy, Matplotlib, PyYAML, TensorBoard und tqdm. Damit sind die Bibliotheken "
            "für Simulation, Training und Visualisierung abgedeckt.",
            st["body"],
        ),
        p(
            "conda env create -f environment.yml<br/>"
            "conda activate holophd<br/>"
            "python -m src.train --config configs/base.yaml",
            st["code"],
        ),
        p(
            "Empfehlung: <font name='DocMono'>python -m src.train</font> ist robuster als "
            "<font name='DocMono'>python src/train.py</font>, weil der Projektwurzelpfad "
            "bei Modulstart zuverlässig für die <font name='DocMono'>src.*</font>-Importe "
            "verfügbar ist.",
            st["callout"],
        ),
        p("3.2 Parameter in base.yaml", st["h2"]),
        table(
            [
                ["Parameter", "Wert", "Wirkung"],
                ["image_size", "128", "Quadratische Bildkante; passend zum CNN-FC-Eingang"],
                ["distance_min/max", "0,95 / 1,05 m", "Symmetrischer Simulationsbereich um 1,00 m"],
                ["num_samples", "3000", "Gesamtzahl synthetischer Hologramme"],
                ["test_split", "0,2", "80 % Training, 20 % Test"],
                ["batch_size", "32", "Beispiele pro Optimierungsschritt"],
                ["epochs", "100", "Vollständige Trainingsdurchläufe"],
                ["learning_rate", "0,001", "Schrittweite des Adam-Optimierers"],
                ["conv_channels", "16, 32, 64, 128", "Featurebreite der vier CNN-Blöcke"],
                ["fc_units", "256", "Breite der dichten Regressionsschicht"],
                ["dropout_rate", "0,3", "Regularisierung vor der Ausgabe"],
            ],
            [38 * mm, 32 * mm, 86 * mm],
            font_size=7.8,
        ),
        p(
            "Die Pfade <font name='DocMono'>data_dir</font> und "
            "<font name='DocMono'>log_dir</font> sind konfiguriert, werden im aktuellen "
            "Trainingsskript jedoch noch nicht verwendet. Modell- und Figurenpfad werden "
            "dagegen aktiv genutzt.",
            st["callout"],
        ),
        PageBreak(),
    ]

    story += [
        p("4. Synthetische Hologrammerzeugung", st["h1"]),
        p(
            "Die Datei <font name='DocMono'>src/data/generator.py</font> enthält drei "
            "Funktionen. Gemeinsam erzeugen sie einen überwachten Datensatz, bei dem der "
            "wahre Ausbreitungsabstand aus der Simulation bekannt ist.",
            st["body"],
        ),
        p("4.1 create_sphere_phantom", st["h2"]),
        p(
            "Ein kreisförmiges, rein phasenschiebendes Objekt wird auf einem quadratischen "
            "Raster erzeugt. Im Zentrum ist die Phase maximal; bis zum zufälligen Radius "
            "fällt sie linear auf null. Die komplexe Transmission lautet "
            "<font name='DocMono'>exp(i * phase)</font>. Da die Amplitude eins bleibt, "
            "entstehen Kontraste erst durch die anschließende Wellenausbreitung.",
            st["body"],
        ),
        table(
            [
                ["Zufallsgröße", "Bereich", "Zweck"],
                ["Radius", "15 bis 29 Pixel", "Variation der Objektgröße"],
                ["Phasenhub", "1,5 bis 3,0 rad", "Variation der Objektstärke"],
                ["Distanz", "0,95 bis 1,05 m", "Lernziel und Fokusvariation"],
            ],
            [46 * mm, 42 * mm, 68 * mm],
        ),
        p("4.2 fresnel_propagate", st["h2"]),
        p(
            "Die Funktion berechnet die räumlichen Frequenzachsen mit "
            "<font name='DocMono'>np.fft.fftfreq</font>, bildet ein zweidimensionales "
            "Frequenzgitter, transformiert das Objektfeld per FFT, multipliziert es mit "
            "einem quadratischen Phasenkern und transformiert zurück. Die Intensität am "
            "Detektor ist anschließend der Betrag zum Quadrat des komplexen Feldes.",
            st["body"],
        ),
        p(
            "field_fft = FFT2(exit_wave)<br/>"
            "propagated_fft = field_fft * kernel<br/>"
            "detector_wave = IFFT2(propagated_fft)<br/>"
            "hologram = abs(detector_wave) ** 2",
            st["code"],
        ),
        p("4.3 generate_dataset", st["h2"]),
        p(
            "Für jedes Beispiel werden Distanz, Radius und Phasenhub unabhängig gezogen. "
            "Jedes Intensitätsbild wird durch sein Maximum geteilt. Das Ziel ist "
            "<font name='DocMono'>(distance - 1.0) * 100</font>: 1,02 m entspricht +2 cm, "
            "0,98 m entspricht -2 cm. Rückgabewerte sind Float32-Arrays für effiziente "
            "Weiterverarbeitung mit PyTorch.",
            st["body"],
        ),
        p(
            "Wissenschaftlich wichtig: Die aktuelle Propagationsformel ist als einfache "
            "Simulation bezeichnet. Vor quantitativen Aussagen muss die Skalierung des "
            "Transferkerns gegen eine etablierte Fresnel- oder Angular-Spectrum-Formulierung "
            "und gegen reale Messgeometrie geprüft werden.",
            st["callout"],
        ),
        PageBreak(),
    ]

    story += [
        p("5. CNN-Architektur", st["h1"]),
        p(
            "<font name='DocMono'>AutofocusCNN</font> ist ein Regressionsnetz. Es gibt "
            "keine Klassenwahrscheinlichkeit aus, sondern einen kontinuierlichen, "
            "signierten Defokuswert in Zentimetern.",
            st["body"],
        ),
        PipelineDiagram(
            [
                "Input\n1 x 128 x 128",
                "Block 1\n16 x 64 x 64",
                "Block 2\n32 x 32 x 32",
                "Block 3\n64 x 16 x 16",
                "Block 4\n128 x 8 x 8",
            ]
        ),
        p("Abbildung 2: Formänderung durch die vier Faltungsblöcke.", st["caption"]),
        table(
            [
                ["Stufe", "Operationen", "Ausgabeform ohne Batch"],
                ["Eingabe", "Normalisiertes Hologramm", "1 x 128 x 128"],
                ["Block 1", "Conv 3x3, BN, ReLU, MaxPool 2", "16 x 64 x 64"],
                ["Block 2", "Conv 3x3, BN, ReLU, MaxPool 2", "32 x 32 x 32"],
                ["Block 3", "Conv 3x3, BN, ReLU, MaxPool 2", "64 x 16 x 16"],
                ["Block 4", "Conv 3x3, BN, ReLU, MaxPool 2", "128 x 8 x 8"],
                ["Flatten", "128 * 8 * 8", "8192"],
                ["Dense", "Linear, ReLU, Dropout 0,3", "256"],
                ["Ausgabe", "Linear", "1 Skalar in cm"],
            ],
            [30 * mm, 76 * mm, 50 * mm],
            font_size=7.8,
        ),
        p("5.1 Bedeutung der Bausteine", st["h2"]),
        bullet("<b>Conv2d:</b> lernt lokale Kanten-, Textur- und Interferenzmuster.", st),
        bullet("<b>BatchNorm2d:</b> stabilisiert Aktivierungsverteilungen im Training.", st),
        bullet("<b>ReLU:</b> führt Nichtlinearität ein und ermöglicht komplexe Merkmale.", st),
        bullet("<b>MaxPool2d:</b> halbiert die räumliche Auflösung und verdichtet Merkmale.", st),
        bullet("<b>Dropout:</b> reduziert Co-Adaption in der voll verbundenen Schicht.", st),
        p("5.2 Feste Eingabegröße", st["h2"]),
        p(
            "Die erste Linear-Schicht erwartet exakt "
            "<font name='DocMono'>channels[3] * 8 * 8</font>. Deshalb funktioniert die "
            "aktuelle Architektur nur direkt mit 128 x 128 Pixeln und vier Pooling-Stufen. "
            "Für flexible Bildgrößen wäre adaptives Pooling vor dem Flatten-Schritt sinnvoll.",
            st["callout"],
        ),
        PageBreak(),
    ]

    story += [
        p("6. Trainingsablauf", st["h1"]),
        p(
            "Die Funktion <font name='DocMono'>main</font> in "
            "<font name='DocMono'>src/train.py</font> orchestriert den kompletten Lauf. "
            "Die folgenden neun Schritte entsprechen direkt der Struktur im Quellcode.",
            st["body"],
        ),
        table(
            [
                ["Nr.", "Schritt", "Technische Wirkung"],
                ["1", "YAML laden", "Parameter werden als Python-Dictionary verfügbar."],
                ["2", "Daten generieren", "X: [3000,128,128], y: [3000]."],
                ["3", "Kanal + Split", "X wird [3000,1,128,128]; 2400/600 Beispiele."],
                ["4", "DataLoader", "Batches zu 32; Training wird gemischt."],
                ["5", "Modell/Optimierer", "CNN, Adam und MSELoss werden erzeugt."],
                ["6", "100 Epochen", "Gradientenupdate im Training; Test ohne Gradienten."],
                ["7", "Endauswertung", "Vorhersagen sammeln und MAE in mm berechnen."],
                ["8", "Grafiken", "Loss-Verlauf und True-vs-Predicted speichern."],
                ["9", "Gewichte", "state_dict als model_final.pth speichern."],
            ],
            [12 * mm, 42 * mm, 102 * mm],
            font_size=7.8,
        ),
        p("6.1 Optimierung", st["h2"]),
        p(
            "Im Trainingsmodus aktualisiert BatchNorm seine Statistik und Dropout ist "
            "aktiv. Pro Batch werden Gradienten gelöscht, Vorhersagen berechnet, der "
            "mittlere quadratische Fehler bestimmt, rückpropagiert und die Parameter mit "
            "Adam angepasst. Im Testmodus deaktiviert "
            "<font name='DocMono'>torch.no_grad()</font> die Gradientenberechnung.",
            st["body"],
        ),
        p(
            "optimizer.zero_grad()<br/>"
            "preds = model(imgs)<br/>"
            "loss = MSE(preds, labels)<br/>"
            "loss.backward()<br/>"
            "optimizer.step()",
            st["code"],
        ),
        p("6.2 Verlustmittelung", st["h2"]),
        p(
            "Der Code mittelt aktuell die bereits batchweise gemittelten Loss-Werte über "
            "die Anzahl der Batches. Bei einem kleineren letzten Batch erhält dieser das "
            "gleiche Gewicht wie volle Batches. Exakter wäre eine mit der Batchgröße "
            "gewichtete Summe, geteilt durch die Zahl aller Beispiele.",
            st["callout"],
        ),
        p("6.3 Hardware", st["h2"]),
        p(
            "Modell und Tensoren werden nicht auf CUDA verschoben. Das Training läuft "
            "daher im aktuellen Zustand auf der CPU, selbst wenn eine GPU verfügbar ist. "
            "Für GPU-Nutzung müssen Gerät, Modell und beide Batch-Tensoren konsistent mit "
            "<font name='DocMono'>.to(device)</font> übertragen werden.",
            st["body"],
        ),
        PageBreak(),
    ]

    story += [
        p("7. Metriken und Visualisierungen", st["h1"]),
        p("7.1 Mean Absolute Error", st["h2"]),
        p(
            "<font name='DocMono'>calculate_mae</font> bildet den Mittelwert der absoluten "
            "Differenzen. Da das CNN Zentimeterabweichungen ausgibt, rechnet "
            "<font name='DocMono'>train.py</font> beide Arrays zunächst auf absolute Meter "
            "zurück und multipliziert den MAE mit 1000. Das Endergebnis wird somit in "
            "Millimetern ausgegeben.",
            st["body"],
        ),
        p(
            "prediction_m = prediction_cm / 100 + 1.0<br/>"
            "truth_m = truth_cm / 100 + 1.0<br/>"
            "MAE_mm = mean(abs(prediction_m - truth_m)) * 1000",
            st["code"],
        ),
        p("7.2 Loss-Kurve", st["h2"]),
        p(
            "<font name='DocMono'>plot_loss_curves</font> zeichnet Trainings- und "
            "Test-MSE je Epoche. Sinkende Kurven zeigen Optimierungsfortschritt. Ein weiter "
            "sinkender Trainings-Loss bei steigendem Test-Loss wäre ein typisches Zeichen "
            "für Overfitting.",
            st["body"],
        ),
        p("7.3 Streudiagramm", st["h2"]),
        p(
            "<font name='DocMono'>plot_predictions</font> stellt wahre und geschätzte "
            "Zentimeterabweichungen gegenüber. Die rote Diagonale markiert perfekte "
            "Vorhersagen. Systematische Krümmung, Offset oder zunehmende Streuung an den "
            "Rändern liefern Hinweise auf Bias, Unteranpassung oder Heteroskedastizität.",
            st["body"],
        ),
        table(
            [
                ["Artefakt", "Pfad", "Interpretation"],
                ["Loss-Kurve", "reports/figures/loss_curve.png", "Konvergenz und Overfitting"],
                ["Scatterplot", "reports/figures/scatter_plot.png", "Bias, Streuung und Randfehler"],
                ["Modell", "reports/trained_models/model_final.pth", "Gelernte PyTorch-Parameter"],
            ],
            [34 * mm, 72 * mm, 50 * mm],
        ),
        p(
            "Die Plotfunktionen rufen nach dem Speichern <font name='DocMono'>plt.show()</font> "
            "auf. In einer Umgebung ohne grafische Oberfläche kann dies blockieren. Für "
            "automatisierte Experimente sollte nach dem Speichern stattdessen die Figur "
            "geschlossen oder ein nicht-interaktives Matplotlib-Backend verwendet werden.",
            st["callout"],
        ),
        PageBreak(),
    ]

    story += [
        p("8. Ausführung und erwartete Resultate", st["h1"]),
        p("8.1 Voraussetzungen", st["h2"]),
        bullet("Miniconda oder Anaconda ist installiert und im Terminal verfügbar.", st),
        bullet("Ausreichender Arbeitsspeicher für simulierte Arrays und Tensor-Kopien.", st),
        bullet("Der Befehl wird aus dem Projektwurzelverzeichnis ausgeführt.", st),
        p("8.2 Empfohlene Befehlsfolge", st["h2"]),
        p(
            "cd C:\\Users\\Moham\\Desktop\\Master\\master_thesis_autofocus<br/>"
            "conda env create -f environment.yml<br/>"
            "conda activate holophd<br/>"
            "python -m src.train --config configs/base.yaml",
            st["code"],
        ),
        p("8.3 Erwarteter Ablauf", st["h2"]),
        bullet("3000 Hologramme werden bei jedem Lauf neu im Arbeitsspeicher erzeugt.", st),
        bullet("Das Terminal zeigt alle zehn Epochen Training- und Test-MSE.", st),
        bullet("Nach 100 Epochen wird der MAE in Millimetern ausgegeben.", st),
        bullet("Zwei PNG-Abbildungen und eine PTH-Gewichtsdatei werden gespeichert.", st),
        p("8.4 Fehlerdiagnose", st["h2"]),
        table(
            [
                ["Symptom", "Wahrscheinliche Ursache", "Maßnahme"],
                ["conda nicht gefunden", "Conda fehlt oder PATH ist nicht initialisiert", "Miniconda installieren; Terminal neu öffnen"],
                ["No module named src", "Skript direkt statt als Modul gestartet", "python -m src.train verwenden"],
                ["Linear shape mismatch", "image_size wurde von 128 geändert", "CNN adaptiv machen oder FC-Dimension anpassen"],
                ["Plot blockiert", "Interaktives Backend / plt.show()", "Agg-Backend nutzen und plt.close() aufrufen"],
                ["Speicherfehler", "Datensatz vollständig im RAM", "On-the-fly Dataset oder kleinere Datenmenge"],
            ],
            [38 * mm, 58 * mm, 60 * mm],
            font_size=7.5,
        ),
        PageBreak(),
    ]

    story += [
        p("9. Wissenschaftliche und technische Grenzen", st["h1"]),
        p(
            "Der aktuelle Stand eignet sich als Startpunkt und Pipeline-Test. Für belastbare "
            "Ergebnisse einer Masterarbeit müssen jedoch mehrere Punkte kontrolliert und "
            "dokumentiert werden.",
            st["body"],
        ),
        table(
            [
                ["Priorität", "Grenze", "Warum relevant"],
                ["Hoch", "Propagationskern validieren", "Falsche physikalische Skalierung macht Lernergebnisse bedeutungslos."],
                ["Hoch", "Unabhängiges Validierungs- und Testset", "Das Testset wird derzeit in jeder Epoche beobachtet."],
                ["Hoch", "Reale Hologramme / Domain Gap", "Ein Netz auf idealisierten Phantomen muss nicht auf Messdaten generalisieren."],
                ["Mittel", "Zufallssamen setzen", "Ohne Seeds sind Datensatz, Split und Training nicht exakt reproduzierbar."],
                ["Mittel", "Baseline-Vergleich", "Geschwindigkeit und Fehler müssen gegen Nelder-Mead und Fokusmetriken gemessen werden."],
                ["Mittel", "Unsicherheit", "Eine Punktschätzung zeigt nicht, wann das Modell unzuverlässig ist."],
                ["Mittel", "GPU und Streaming", "Aktueller CPU-/RAM-Ansatz skaliert schlecht."],
            ],
            [24 * mm, 52 * mm, 80 * mm],
            font_size=7.5,
        ),
        p("9.1 Datenleckage und Modellauswahl", st["h2"]),
        p(
            "Da der sogenannte Test-Loss nach jeder Epoche berechnet wird, beeinflusst er "
            "typischerweise Entwicklungsentscheidungen. Methodisch sollte es drei getrennte "
            "Mengen geben: Training für Gradienten, Validierung für Hyperparameter und "
            "frühes Stoppen sowie ein bis zum Schluss unangetastetes Testset.",
            st["body"],
        ),
        p("9.2 Sim-to-real-Transfer", st["h2"]),
        p(
            "Die synthetischen Beispiele variieren nur Radius, Phasenhub und Distanz. Reale "
            "Daten enthalten zusätzlich Rauschen, Detektorantwort, Flat-Field-Fehler, "
            "unbekannte Objektformen, partielle Kohärenz, Strahlprofil und Rekonstruktionsartefakte. "
            "Diese Faktoren müssen simuliert, augmentiert oder durch reale Trainingsdaten "
            "abgedeckt werden.",
            st["body"],
        ),
        p("9.3 Zieldefinition", st["h2"]),
        p(
            "Das Modell sagt derzeit eine globale Distanz für ein gesamtes Hologramm voraus. "
            "Für geneigte Proben, räumlich variierende Defokussierung oder mehrere Tiefen "
            "wäre ein skalares Ziel unzureichend. Dann wären Patch-Regression, Tiefenkarten "
            "oder probabilistische Ausgaben zu prüfen.",
            st["body"],
        ),
        PageBreak(),
    ]

    story += [
        p("10. Empfohlener Entwicklungsplan", st["h1"]),
        table(
            [
                ["Phase", "Ziel", "Abnahmekriterium"],
                ["1. Physik", "Propagation mit Referenzimplementierung prüfen", "Numerische Tests und dimensionskonsistente Gleichungen"],
                ["2. Reproduzierbarkeit", "Seeds, gespeicherte Splits, Metadaten", "Identische Läufe liefern identische Resultate"],
                ["3. Evaluation", "Train/Val/Test und Baselines", "Vorab definierte MAE- und Laufzeitmetriken"],
                ["4. Datenrealismus", "Rauschen und Systemeffekte", "Robustheit über kontrollierte Störgrade"],
                ["5. Reale Daten", "Transfer auf Messhologramme", "Vergleich mit Referenzfokus und Expertenprüfung"],
                ["6. Ablation", "Architektur- und Datenfaktoren isolieren", "Beitrag jeder Komponente quantitativ belegt"],
                ["7. Deployment", "Schnelle Inferenzpipeline", "Latenz und Speicherverbrauch dokumentiert"],
            ],
            [30 * mm, 62 * mm, 64 * mm],
            font_size=7.5,
        ),
        p("10.1 Sinnvolle Code-Erweiterungen", st["h2"]),
        bullet("PyTorch-Dataset erzeugt Hologramme bei Bedarf statt alles im RAM zu halten.", st),
        bullet("Explizites Device-Handling für CPU, CUDA und gegebenenfalls MPS.", st),
        bullet("Seed-Funktion für Python, NumPy, PyTorch und DataLoader-Worker.", st),
        bullet("Validation-Loop, Early Stopping und Checkpoint des besten Modells.", st),
        bullet("TensorBoard-Logging, obwohl TensorBoard bereits als Abhängigkeit vorhanden ist.", st),
        bullet("Konfigurierbare Pixelgröße, Wellenlänge und Referenzdistanz in YAML.", st),
        bullet("Unit-Tests für Formen, Einheiten, Propagation und Metriken.", st),
        p("10.2 Empfohlene Kennzahlen", st["h2"]),
        bullet("MAE, RMSE und 95. Perzentil des absoluten Fehlers in mm.", st),
        bullet("Bias als mittlerer signierter Fehler über die Distanzspanne.", st),
        bullet("Fehler getrennt nach Distanz, Objektgröße, Phasenhub und Rauschgrad.", st),
        bullet("Inferenzzeit pro Hologramm und Speed-up gegenüber Nelder-Mead.", st),
        bullet("Kalibrierung oder Intervallabdeckung bei probabilistischen Modellen.", st),
        PageBreak(),
    ]

    story += [
        p("11. Datei-für-Datei-Referenz", st["h1"]),
        table(
            [
                ["Datei", "Zentrale Symbole", "Kurzbeschreibung"],
                ["configs/base.yaml", "Schlüssel/Werte", "Experimentparameter und Ausgabepfade"],
                ["src/data/generator.py", "fresnel_propagate", "FFT-basierte Wellenpropagation"],
                ["", "create_sphere_phantom", "Kreisförmiges komplexes Phasenobjekt"],
                ["", "generate_dataset", "Erzeugt X-Hologramme und y-Defokuslabels"],
                ["src/models/cnn.py", "AutofocusCNN", "Vier CNN-Blöcke plus Regressionskopf"],
                ["src/utils/metrics.py", "calculate_mae", "Mittlerer absoluter Fehler"],
                ["src/utils/plotting.py", "plot_loss_curves", "Trainings- und Test-Loss"],
                ["", "plot_predictions", "True-vs-Predicted-Streudiagramm"],
                ["src/train.py", "main", "Orchestriert vollständiges Experiment"],
                ["environment.yml", "holophd", "Conda-Laufzeitumgebung"],
                [".gitignore", "Muster", "Schließt Daten und generierte Artefakte aus"],
            ],
            [46 * mm, 48 * mm, 62 * mm],
            font_size=7.4,
        ),
        p("11.1 Datenformen im Gesamtablauf", st["h2"]),
        table(
            [
                ["Variable", "Form", "Datentyp"],
                ["X_raw vor Kanal", "[3000, 128, 128]", "float32"],
                ["X_raw nach Kanal", "[3000, 1, 128, 128]", "float32"],
                ["X_train / X_test", "[2400,...] / [600,...]", "float32"],
                ["y_train / y_test", "[2400] / [600]", "float32, cm"],
                ["CNN-Ausgabe", "[Batch]", "float32, cm"],
                ["model_final.pth", "state_dict", "Parameter-Tensoren"],
            ],
            [50 * mm, 58 * mm, 48 * mm],
        ),
        p("11.2 Schlussfolgerung", st["h2"]),
        p(
            "Das Projekt besitzt eine klare, modular erweiterbare Grundstruktur und bildet "
            "den vollständigen Weg von einer kontrollierten Simulation zu einer "
            "lernenden Fokusregression ab. Der größte nächste Schritt ist nicht eine "
            "komplexere CNN-Architektur, sondern die physikalische Validierung der "
            "Datenerzeugung und ein methodisch sauberes Evaluationsdesign. Erst danach "
            "lassen sich Architekturvergleiche und der behauptete Geschwindigkeitsvorteil "
            "gegenüber iterativen Autofokusverfahren wissenschaftlich belastbar bewerten.",
            st["body"],
        ),
        Spacer(1, 10 * mm),
        HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CAD8DF")),
        Spacer(1, 4 * mm),
        p(
            "Dokumentationsbasis: aktueller Quellcode im lokalen Projektordner "
            "master_thesis_autofocus. Externe Fachliteratur wurde in diesem Dokument nicht "
            "inhaltlich ausgewertet.",
            st["small"],
        ),
    ]
    return story


def build_pdf() -> Path:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=21 * mm,
        rightMargin=21 * mm,
        topMargin=21 * mm,
        bottomMargin=18 * mm,
        title="Projektdokumentation - Learning-based Autofocus for Holography",
        author="Codex",
        subject="Technische Dokumentation des Masterarbeitsprojekts",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="main",
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[frame], onPage=cover_page, autoNextPageTemplate="body"),
            PageTemplate(id="body", frames=[frame], onPage=header_footer),
        ]
    )
    doc.build(build_story(styles()))
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
