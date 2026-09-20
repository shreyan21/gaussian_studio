[CmdletBinding()]
param(
    [switch]$SkipColmapDownload,
    [string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program $($Arguments -join ' ')" }
}

Write-Host 'Gaussian Scene Studio - pretrained-free setup' -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    $candidates = @()
    if ($PythonExe) { $candidates += $PythonExe }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $found = & $launcher.Source -3.11 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $candidates += $found }
        } catch { }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python -and $python.Source -notlike '*\WindowsApps\*') { $candidates += $python.Source }
    $basePython = $null
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        & $candidate -c 'import sys; assert sys.version_info[:2] == (3,11) and sys.maxsize > 2**32' 2>$null
        if ($LASTEXITCODE -eq 0) { $basePython = $candidate; break }
    }
    if (-not $basePython) { throw 'Install 64-bit Python 3.11, open a new PowerShell window, then rerun setup.' }
    Invoke-Checked $basePython @('-m','venv','.venv')
}

$appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Invoke-Checked $appPython @('-m','pip','install','--upgrade','pip')
Invoke-Checked $appPython @('-m','pip','install','-r','requirements.txt')

$colmapRoot = Join-Path $PSScriptRoot 'tools\colmap'
$colmap = Get-ChildItem -LiteralPath $colmapRoot -Filter 'COLMAP.bat' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $colmap -and -not $SkipColmapDownload) {
    $tools = Join-Path $PSScriptRoot 'tools'
    $archive = Join-Path $tools 'colmap-4.2.0-windows-cuda.zip'
    New-Item -ItemType Directory -Path $tools -Force | Out-Null
    New-Item -ItemType Directory -Path $colmapRoot -Force | Out-Null
    Write-Host 'Downloading official COLMAP 4.2.0 CUDA package (~381 MB)...' -ForegroundColor Cyan
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/colmap/colmap/releases/download/4.2.0/colmap-x64-windows-cuda.zip' -OutFile $archive
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    $expected = '991e0bae403a496fcc4de0c1f1f428619bf12f8000978f77bc6799d9bfeac23e'
    if ($actual -ne $expected) { throw "COLMAP checksum mismatch. Expected $expected, got $actual." }
    Expand-Archive -LiteralPath $archive -DestinationPath $colmapRoot -Force
    Remove-Item -LiteralPath $archive -Force
    $colmap = Get-ChildItem -LiteralPath $colmapRoot -Filter 'COLMAP.bat' -File -Recurse | Select-Object -First 1
}
if (-not $colmap) { throw 'CUDA COLMAP is missing. Rerun without -SkipColmapDownload or set GSS_COLMAP to COLMAP.bat.' }

Invoke-Checked $appPython @('-m','pip','check')
Invoke-Checked $appPython @('scripts\doctor.py','--require-custom')
Invoke-Checked $appPython @('-m','pytest','-q','tests')
Write-Host 'Setup complete. Double-click Start Studio.cmd.' -ForegroundColor Green
