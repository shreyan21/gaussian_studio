[CmdletBinding()]
param([ValidateRange(1024,65535)][int]$Port = 7860)

$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $appRoot
$python = Join-Path $appRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Run Setup NVIDIA Workstation.cmd first.'
}

$toolDir = Join-Path $appRoot 'tools'
$cloudflared = Join-Path $toolDir 'cloudflared.exe'
if (-not (Test-Path -LiteralPath $cloudflared)) {
    New-Item -ItemType Directory -Force -Path $toolDir | Out-Null
    $download = "$cloudflared.download"
    Write-Host 'Downloading official Cloudflare Tunnel client...' -ForegroundColor Cyan
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile $download
    Move-Item -LiteralPath $download -Destination $cloudflared -Force
}

$tokenBytes = New-Object byte[] 24
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
try { $rng.GetBytes($tokenBytes) } finally { $rng.Dispose() }
$accessToken = -join ($tokenBytes | ForEach-Object { $_.ToString('x2') })
$env:GSS_ACCESS_TOKEN = $accessToken

$logDir = Join-Path $appRoot 'data\public-link'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$serverOut = Join-Path $logDir 'server.out.log'
$serverErr = Join-Path $logDir 'server.err.log'
$tunnelOut = Join-Path $logDir 'tunnel.out.log'
$tunnelErr = Join-Path $logDir 'tunnel.err.log'

$server = $null
$tunnel = $null
try {
    $server = Start-Process -FilePath $python -ArgumentList @('run.py','--no-browser','--port',"$Port") -WorkingDirectory $appRoot -WindowStyle Hidden -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr -PassThru
    $health = "http://127.0.0.1:$Port/api/health?token=$accessToken"
    $ready = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        if ($server.HasExited) { throw "Studio stopped during startup. See $serverErr" }
        try {
            $status = Invoke-RestMethod -Uri $health -TimeoutSec 2
            if ($status.app -eq 'Gaussian Scene Studio') { $ready = $true; break }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $ready) { throw 'Studio did not become ready within 45 seconds.' }

    $tunnel = Start-Process -FilePath $cloudflared -ArgumentList @('tunnel','--url',"http://127.0.0.1:$Port",'--no-autoupdate') -WorkingDirectory $appRoot -WindowStyle Hidden -RedirectStandardOutput $tunnelOut -RedirectStandardError $tunnelErr -PassThru
    $publicUrl = $null
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($tunnel.HasExited) { throw "Tunnel stopped during startup. See $tunnelErr" }
        $logText = ((Get-Content -LiteralPath $tunnelOut -Raw -ErrorAction SilentlyContinue) + "`n" + (Get-Content -LiteralPath $tunnelErr -Raw -ErrorAction SilentlyContinue))
        $match = [regex]::Match($logText, 'https://[a-z0-9-]+\.trycloudflare\.com')
        if ($match.Success) { $publicUrl = $match.Value; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $publicUrl) { throw "Tunnel URL not found. See $tunnelErr" }

    $protectedUrl = "$publicUrl/?token=$accessToken"
    Write-Host ''
    Write-Host 'Protected live link:' -ForegroundColor Green
    Write-Host $protectedUrl
    Write-Host ''
    Write-Host 'Keep this window open. Press Ctrl+C to stop app and public link.' -ForegroundColor Yellow
    Start-Process $protectedUrl
    while (-not $server.HasExited -and -not $tunnel.HasExited) { Start-Sleep -Seconds 1 }
    if ($server.HasExited) { throw "Studio stopped. See $serverErr" }
    throw "Tunnel stopped. See $tunnelErr"
}
finally {
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue }
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue }
}
