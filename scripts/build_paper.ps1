$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PaperDir = Join-Path $Root "paper"
$FinalDir = Join-Path $PaperDir "final"
$DownloadsPdf = Join-Path $HOME "Downloads\iclr_submission_bon_mcts_dynamics.pdf"

New-Item -ItemType Directory -Force $FinalDir | Out-Null

foreach ($TemplateFile in @("iclr2026_conference.sty", "iclr2026_conference.bst")) {
    $Path = Join-Path $PaperDir $TemplateFile
    if (-not (Test-Path $Path)) {
        throw "Missing required ICLR template file: $Path"
    }
}

Push-Location $PaperDir
try {
    Remove-Item -Force "main.aux", "main.bbl", "main.blg", "main.log", "main.out" -ErrorAction SilentlyContinue
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
