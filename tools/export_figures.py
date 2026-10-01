#!/usr/bin/env python3
"""Exportiert Abbildungen aus einem Trainingslauf in die Thesis.

Kopiert ausgewählte Figuren aus ``runs/<name>/`` (rekursiv gesucht, Formate
PDF/PNG/JPG/EPS) nach ``thesis/figures/<kapitel>/`` und schreibt je Figur ein
LaTeX-Snippet ``thesis/figures/<kapitel>/<name>.tex`` mit ``\\includegraphics``,
Caption-Platzhalter, Label ``fig:<kapitel>:<name>`` und einem Kommentar, der die
Quelle festhält (Run-Pfad, Commit-SHA, Datum, Dateien). Im Kapitel genügt dann

    \\input{figures/<kapitel>/<name>}

Figurnamen, die ``src.train`` bzw. ``src.evaluate`` erzeugen (Dateistamm = Name):

    runs/<run>/loss_curves.png                      Lernkurven (Train/Val-Loss, Lernrate)
    runs/<run>/test_scatter_target.png              Zielraum (z. B. ln Fr) wahr vs. vorhergesagt
    runs/<run>/test_scatter_z01_mm.png              z01 in mm wahr vs. vorhergesagt
    runs/<run>/test_error_vs_z01.png                Fehler über z01 (gebinnt)
    runs/<run>/test_examples.png                    Beispiel-Hologramme des Testsplits
    runs/<run>/eval_<split>/{scatter_target,scatter_z01_mm,error_vs_z01,examples}.png
                                                    dasselbe aus ``src.evaluate`` (Standard: eval_test/)

Beispiele::

    python tools/export_figures.py --run runs/2026-10-15_cnn-baseline_s0 --list
    python tools/export_figures.py --run runs/2026-10-15_cnn-baseline_s0 \\
        --chapter experimente --names loss_curves test_scatter_z01_mm
    python tools/export_figures.py --run runs/x --chapter anhang --names "test_*"   # alle Test-Plots
    python tools/export_figures.py --run runs/x/eval_ood --chapter anhang --names "*"   # ein eval_*-Ordner

Bestehende ``.tex``-Snippets werden nicht überschrieben (die Caption darin ist
Handarbeit), es sei denn ``--force`` ist gesetzt; die Bilddateien werden immer
aktualisiert. Liegt derselbe Dateiname in mehreren Unterordnern (z. B.
``eval_test/`` und ``eval_ood/``), wird die Figur übersprungen – dann ``--run``
auf den gewünschten Unterordner zeigen lassen. Nur Standardbibliothek; Git wird
ausschließlich lesend benutzt (``git rev-parse HEAD``, ``git status --porcelain``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import re
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".eps")
# Reihenfolge, in der LaTeX bei \includegraphics{name} ohne Endung sucht: PDF zuerst.
SUFFIX_PRIORITY = {s: i for i, s in enumerate(IMAGE_SUFFIXES)}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "thesis" / "figures"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

SNIPPET_TEMPLATE = """% ---------------------------------------------------------------------------
% Automatisch erzeugt von tools/export_figures.py -- Quelle der Abbildung:
%   Run:     {run}
%   Commit:  {commit}{dirty}
%   Datum:   {date}
%   Dateien: {files}
% Caption und ggf. Breite anpassen; diese Datei wird bei erneutem Export NICHT
% überschrieben (außer mit --force). Einbinden im Kapitel:
%   \\input{{figures/{chapter}/{name}}}
% ---------------------------------------------------------------------------
\\begin{{figure}}[{placement}]
  \\centering
  \\includegraphics[width={width}]{{figures/{chapter}/{name}}}
  \\caption[{short_caption}]{{TODO: Caption für \\texttt{{{name_tex}}} (Run \\texttt{{{run_tex}}}).}}
  \\label{{fig:{chapter}:{name}}}
\\end{{figure}}
"""


def _run_git(args: list[str]) -> str | None:
    """Führt einen lesenden Git-Befehl aus; None, wenn Git fehlt oder scheitert."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_commit_info() -> tuple[str, bool]:
    """(Commit-SHA oder 'unbekannt', working tree dirty?)."""
    sha = _run_git(["rev-parse", "HEAD"]) or "unbekannt (kein Git-Repository/kein Commit)"
    status = _run_git(["status", "--porcelain", "--untracked-files=no"])
    dirty = bool(status)
    return sha, dirty


def find_figures(run_dir: Path) -> dict[str, list[Path]]:
    """Alle Bilddateien unterhalb von run_dir, gruppiert nach Dateistamm."""
    found: dict[str, list[Path]] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found.setdefault(path.stem, []).append(path)
    for stem, paths in found.items():
        paths.sort(key=lambda p: (SUFFIX_PRIORITY[p.suffix.lower()], str(p)))
    return found


def ambiguous_sources(paths: list[Path]) -> list[Path]:
    """Dateien eines Stamms, deren Endung mehrfach vorkommt (gleicher Name in mehreren Ordnern)."""
    seen: dict[str, Path] = {}
    duplicates: list[Path] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in seen:
            if seen[suffix] not in duplicates:
                duplicates.append(seen[suffix])
            duplicates.append(path)
        else:
            seen[suffix] = path
    return duplicates


def select_names(available: dict[str, list[Path]], patterns: list[str]) -> tuple[list[str], list[str]]:
    """Löst Namen/Glob-Muster gegen die verfügbaren Stämme auf."""
    selected: list[str] = []
    missing: list[str] = []
    for pattern in patterns:
        matches = sorted(stem for stem in available if fnmatch.fnmatchcase(stem, pattern))
        if not matches:
            missing.append(pattern)
            continue
        for stem in matches:
            if stem not in selected:
                selected.append(stem)
    return selected, missing


def tex_escape(text: str) -> str:
    """Minimales Escaping für Text in \\texttt{}."""
    return text.replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def export(
    run_dir: Path,
    chapter: str,
    names: list[str],
    figures_dir: Path,
    width: str,
    placement: str,
    force: bool,
    dry_run: bool,
) -> int:
    available = find_figures(run_dir)
    if not available:
        print(f"Keine Bilddateien ({', '.join(IMAGE_SUFFIXES)}) unter {run_dir} gefunden.", file=sys.stderr)
        return 1

    selected, missing = select_names(available, names)
    for pattern in missing:
        print(f"WARNUNG: keine Figur passt auf '{pattern}' in {run_dir} (siehe --list).", file=sys.stderr)
    if not selected:
        print("Nichts zu exportieren.", file=sys.stderr)
        return 1

    for stem in list(selected):
        if not NAME_RE.match(stem):
            print(f"WARNUNG: Name '{stem}' enthält ungünstige Zeichen für LaTeX-Labels/Dateinamen – übersprungen.", file=sys.stderr)
            selected.remove(stem)
            continue
        duplicates = ambiguous_sources(available[stem])
        if duplicates:
            where = ", ".join(relative_to_repo(p) for p in duplicates)
            print(
                f"WARNUNG: '{stem}' liegt mehrfach vor ({where}) – übersprungen; --run auf den gewünschten Unterordner setzen.",
                file=sys.stderr,
            )
            selected.remove(stem)
            missing.append(stem)
    if not selected:
        print("Nichts zu exportieren.", file=sys.stderr)
        return 1

    target_dir = figures_dir / chapter
    sha, dirty = git_commit_info()
    run_label = relative_to_repo(run_dir)
    today = _dt.date.today().isoformat()

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for stem in selected:
        sources = available[stem]
        copied_names = []
        for src in sources:
            dst = target_dir / (stem + src.suffix.lower())
            copied_names.append(dst.name)
            if dry_run:
                print(f"[dry-run] kopiere {relative_to_repo(src)} -> {relative_to_repo(dst)}")
            else:
                shutil.copy2(src, dst)
                print(f"kopiert   {relative_to_repo(src)} -> {relative_to_repo(dst)}")

        snippet_path = target_dir / f"{stem}.tex"
        if snippet_path.exists() and not force:
            print(f"behalten  {relative_to_repo(snippet_path)} (existiert; --force zum Überschreiben)")
        else:
            snippet = SNIPPET_TEMPLATE.format(
                run=run_label,
                commit=sha,
                dirty="  (Working Tree hatte uncommittete Änderungen!)" if dirty else "",
                date=today,
                files=", ".join(relative_to_repo(s) for s in sources),
                chapter=chapter,
                name=stem,
                name_tex=tex_escape(stem),
                run_tex=tex_escape(run_label),
                width=width,
                placement=placement,
                short_caption=stem.replace("_", " "),
            )
            if dry_run:
                print(f"[dry-run] schreibe {relative_to_repo(snippet_path)}")
            else:
                snippet_path.write_text(snippet, encoding="utf-8")
                print(f"snippet   {relative_to_repo(snippet_path)}  -> \\input{{figures/{chapter}/{stem}}}")
        exported += 1

    print(f"\n{exported} Figur(en) exportiert nach {relative_to_repo(target_dir)}  (Commit {sha[:12]}{' +dirty' if dirty else ''}).")
    return 0 if not missing else 2


def list_figures(run_dir: Path) -> int:
    available = find_figures(run_dir)
    if not available:
        print(f"Keine Bilddateien ({', '.join(IMAGE_SUFFIXES)}) unter {run_dir} gefunden.")
        return 1
    print(f"Verfügbare Figuren in {run_dir}:")
    width = max(len(s) for s in available)
    for stem, paths in sorted(available.items()):
        formats = ", ".join(p.suffix.lower().lstrip(".") for p in paths)
        parents = sorted({relative_to_repo(p.parent) for p in paths})
        where = ", ".join(f"{parent}/" for parent in parents)
        note = "  (mehrdeutig: --run auf einen Unterordner setzen)" if ambiguous_sources(paths) else ""
        print(f"  {stem:<{width}}  [{formats}]  {where}{note}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kopiert Figuren aus runs/<name>/ nach thesis/figures/<kapitel>/ und erzeugt LaTeX-Snippets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Beispiele::", 1)[-1] if __doc__ else None,
    )
    parser.add_argument("--run", required=True, type=Path, help="Run-Ordner, z. B. runs/2026-10-15_cnn-baseline_s0")
    parser.add_argument("--list", action="store_true", help="nur verfügbare Figuren anzeigen")
    parser.add_argument("--chapter", help="Zielkapitel, z. B. grundlagen | methoden | experimente | anhang")
    parser.add_argument("--names", nargs="+", default=[], help="Figurnamen (Dateistamm ohne Endung); Glob-Muster wie 'scatter_*' erlaubt")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR, help=f"Zielbasis (Standard: {DEFAULT_FIGURES_DIR})")
    parser.add_argument("--width", default=r"0.85\textwidth", help=r"Breite für \includegraphics (Standard: 0.85\textwidth)")
    parser.add_argument("--placement", default="htbp", help="Float-Platzierung (Standard: htbp)")
    parser.add_argument("--force", action="store_true", help="bestehende .tex-Snippets überschreiben")
    parser.add_argument("--dry-run", action="store_true", help="nur anzeigen, nichts schreiben")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir: Path = args.run
    if not run_dir.is_dir():
        print(f"FEHLER: Run-Ordner '{run_dir}' existiert nicht.", file=sys.stderr)
        return 1

    if args.list:
        return list_figures(run_dir)

    if not args.chapter or not args.names:
        print("FEHLER: --chapter und --names sind erforderlich (oder --list verwenden).", file=sys.stderr)
        return 1
    if not re.match(r"^[a-z0-9][a-z0-9_-]*$", args.chapter):
        print("FEHLER: --chapter nur mit Kleinbuchstaben, Ziffern, '_' oder '-' (wird Teil des Labels).", file=sys.stderr)
        return 1

    return export(
        run_dir=run_dir,
        chapter=args.chapter,
        names=args.names,
        figures_dir=args.figures_dir,
        width=args.width,
        placement=args.placement,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
