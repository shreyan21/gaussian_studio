[CmdletBinding()]
param(
    [ValidateSet('CPU','CUDA')][string]$Device = 'CPU',
    [switch]$SkipModelDownload,
    [switch]$SkipInferenceTest,
    [switch]$IncludeSharp,
    [string]$PythonExe = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program $($Arguments -join ' ')" }
}

Write-Host 'Gaussian Scene Studio setup' -ForegroundColor Cyan
Write-Host "Selected inference runtime: $Device"
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    $candidates = @()
    if ($PythonExe) { $candidates += $PythonExe }
    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $found = & $pyLauncher.Source -3.11 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $candidates += $found }
        } catch { }
    }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notlike '*\WindowsApps\*') { $candidates += $pythonCommand.Source }
    $basePython = $null
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        try {
            & $candidate -c 'import sys; assert sys.version_info[:2] == (3,11) and sys.maxsize > 2**32' 2>$null
            if ($LASTEXITCODE -eq 0) { $basePython = $candidate; break }
        } catch { }
    }
    if (-not $basePython) { throw 'Install 64-bit Python 3.11 from python.org (include the launcher), then run setup again. Or use -PythonExe C:\Path\To\python.exe.' }
    Invoke-Checked $basePython @('-m','venv','.venv')
}
$appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Invoke-Checked $appPython @('-c','import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)')
Invoke-Checked $appPython @('-m','pip','install','--upgrade','pip')
Invoke-Checked $appPython @('-m','pip','install','-r','requirements.txt','-r','requirements-torch-runtime.txt','-c','constraints-windows-py311.txt')
$wheelIndex = if ($Device -eq 'CUDA') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
# A device switch must replace a pre-existing CPU/CUDA wheel, not report it satisfied.
$desiredRuntime = if ($Device -eq 'CUDA') { 'cu128' } else { 'cpu' }
$installedRuntime = & $appPython 'scripts\runtime_version.py'
$torchArguments = @('-m','pip','install','--no-deps','torch==2.8.0','torchvision==0.23.0','--index-url',$wheelIndex)
if ($installedRuntime -notlike "*+$desiredRuntime*") { $torchArguments += '--force-reinstall' }
Invoke-Checked $appPython $torchArguments
if ($Device -eq 'CUDA') {
    Invoke-Checked $appPython @('-m','pip','install','-r','requirements-anysplat.txt','-c','constraints-windows-py311.txt')
    Invoke-Checked $appPython @('scripts\verify_anysplat.py','--check-import')
}
if ($IncludeSharp) {
    Invoke-Checked $appPython @('-m','pip','install','-r','requirements-sharp.txt','-c','constraints-windows-py311.txt')
}
if (-not $SkipModelDownload) {
    $modelChoice = if ($Device -eq 'CUDA') { 'all' } else { 'depth' }
    Invoke-Checked $appPython @('scripts\download_models.py','--model',$modelChoice)
    if ($Device -eq 'CUDA') { Invoke-Checked $appPython @('scripts\verify_anysplat.py','--check-hash','--check-import') }
    if ($IncludeSharp) {
        Write-Host 'SHARP is licensed only for non-commercial scientific research.' -ForegroundColor Yellow
        Invoke-Checked $appPython @('scripts\download_models.py','--model','sharp')
        Invoke-Checked $appPython @('scripts\verify_sharp.py','--check-hash','--check-import')
    }
}
Invoke-Checked $appPython @('-m','pip','check')
$doctorArgs = @('scripts\doctor.py')
if ($Device -eq 'CUDA') {
    $doctorArgs += '--require-cuda'
    if (-not $SkipModelDownload) { $doctorArgs += '--require-anysplat' }
}
Invoke-Checked $appPython $doctorArgs
if ($Device -eq 'CUDA' -and -not $SkipModelDownload -and -not $SkipInferenceTest) {
    Write-Host 'Running one real two-view AnySplat CUDA smoke test...' -ForegroundColor Cyan
    Invoke-Checked $appPython @('scripts\verify_anysplat.py','--check-import','--check-hash','--check-inference')
}
Write-Host 'Setup complete. Double-click Start Studio.cmd.' -ForegroundColor Green
