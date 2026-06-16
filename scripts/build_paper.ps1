param(
    [string]$DesktopCopy = ""
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PaperDir = Join-Path $Root "paper"
$FinalDir = Join-Path $PaperDir "final"
$RepoPdf = Join-Path $FinalDir "best of n mcts tree search learned dynamics-v4.pdf"

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
    Copy-Item -Force "main.pdf" $RepoPdf
    if ($DesktopCopy) {
        $DesktopDir = Split-Path -Parent $DesktopCopy
        if ($DesktopDir) {
            New-Item -ItemType Directory -Force -Path $DesktopDir | Out-Null
        }
        Copy-Item -Force $RepoPdf $DesktopCopy
    }
}
finally {
    Pop-Location
}

Write-Output "Saved $RepoPdf"
if ($DesktopCopy) {
    Write-Output "Saved $DesktopCopy"
}
