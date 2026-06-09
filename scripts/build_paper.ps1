$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PaperDir = Join-Path $Root "paper"
$FinalDir = Join-Path $PaperDir "final"
$DownloadsPdf = Join-Path $HOME "Downloads\iclr_submission_bon_mcts_dynamics.pdf"

New-Item -ItemType Directory -Force $FinalDir | Out-Null

Push-Location $PaperDir
try {
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    Copy-Item -Force "main.pdf" (Join-Path $FinalDir "iclr_submission.pdf")
    Copy-Item -Force (Join-Path $FinalDir "iclr_submission.pdf") $DownloadsPdf
}
finally {
    Pop-Location
}

Write-Output "Saved paper/final/iclr_submission.pdf"
Write-Output "Saved $DownloadsPdf"

