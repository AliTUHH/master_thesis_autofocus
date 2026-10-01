# .latexmkrc -- Build-Konfiguration für latexmk (TeX Live / MiKTeX)
#
#   latexmk            -> baut main.pdf (pdflatex + biber, so oft wie nötig)
#   latexmk -lualatex  -> alternativ mit LuaLaTeX (fontspec-Zweig in main.tex)
#   latexmk -pvc       -> kontinuierlich neu bauen beim Speichern
#   latexmk -C         -> alle Build-Artefakte löschen
#
# Overleaf benutzt ebenfalls latexmk und liest diese Datei automatisch.

$pdf_mode = 1;                 # 1 = pdflatex, 4 = lualatex, 5 = xelatex
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$lualatex = 'lualatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
$xelatex  = 'xelatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';

$bibtex_use = 2;               # biber/bibtex aufrufen und .bbl bei -C mit löschen
$biber = 'biber %O %S';

@default_files = ('main.tex');
$out_dir = '.';

# Artefakte, die latexmk -C zusätzlich entfernen soll (acro, biblatex, synctex)
$clean_ext = 'acn acr alg bbl bcf blg glg glo gls ist loa lol run.xml synctex.gz synctex(busy) nav snm vrb xdv';
