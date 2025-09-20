@echo off
REM ------------------------------------------
REM Konvertiert alle Jupyter Notebooks (*.ipynb)
REM im aktuellen Ordner nach HTML und PDF mit Quarto.
REM Ergebnisse: je eine HTML- und PDF-Datei direkt
REM im selben Ordner wie das Notebook.
REM ------------------------------------------

setlocal enabledelayedexpansion

set INPUT_DIR=%~dp0

echo Starte Konvertierung aller Notebooks in "%INPUT_DIR%"...

for %%F in ("%INPUT_DIR%*.ipynb") do (
    echo ------------------------------------------
    echo Verarbeite %%~nxF ...
    
    REM HTML erzeugen (self-contained)
    quarto render "%%F" --to html --embed-resources --standalone
    
    REM PDF erzeugen
    quarto render "%%F" --pdf-engine=pdflatex  --to pdf 
)

echo ------------------------------------------
echo Fertig! HTML- und PDF-Dateien wurden erstellt.
echo ------------------------------------------

pause
