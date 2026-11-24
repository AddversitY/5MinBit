# Starts the btc5m paper-testing engine on Windows.
#
# Paper mode only: reads public market data and writes predictions to a local, hash-chained
# ledger. It needs no administrator rights, API keys, wallet or money.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run-paper-test.ps1
#   ... -InstallAutostart     also start it (minimized) every time you log in
#   ... -RemoveAutostart      undo -InstallAutostart
#   ... -RebuildDashboard     rebuild the web dashboard (after updating the code)
# Or double-click scripts\windows\start-paper-test.cmd
#
# Dashboard: http://127.0.0.1:8000   Logs: data\logs\serve.log
param(
    [int]$Port = 8000,
    [switch]$InstallAutostart,
    [switch]$RemoveAutostart,
    [switch]$RebuildDashboard,
    [switch]$SetupOnly        # install everything, then exit without starting (used by CI)
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$StartupFile = Join-Path ([Environment]::GetFolderPath("Startup")) "btc5m paper test.cmd"

function Invoke-Checked([string]$what, [scriptblock]$cmd) {
    # Native tools write progress to stderr; judge them by exit code, not by stderr output.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $cmd } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

if ($RemoveAutostart) {
    Remove-Item $StartupFile -ErrorAction SilentlyContinue
    Write-Host "Autostart removed."
    exit 0
}

# Already running? Then do nothing (one server per ledger; the server also enforces this).
try {
    Invoke-WebRequest "http://127.0.0.1:$Port/api/health" -UseBasicParsing -TimeoutSec 3 | Out-Null
    Write-Host "btc5m is already running: http://127.0.0.1:$Port"
    exit 0
} catch { }

# 1. uv: Python package manager, per-user install (official installer from astral.sh)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (per-user, no admin rights needed)..."
    Invoke-Checked "uv install" { powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex" }
}

# 2. Python 3.13 environment with the project installed
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating the Python environment..."
    Invoke-Checked "uv venv" { uv venv --python 3.13 .venv }
}
Write-Host "Checking Python packages..."
Invoke-Checked "package install" { uv pip install --quiet --python .venv\Scripts\python.exe -e . }

# 3. Dashboard (built once with Node.js; a portable, checksum-verified copy is used if none is installed)
if ($RebuildDashboard -or -not (Test-Path "dashboard\dist\index.html")) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
        $NodeVer = "v22.11.0"
        $NodeSha = "905373a059aecaf7f48c1ce10ffbd5334457ca00f678747f19db5ea7d256c236"
        $Tools = Join-Path $Root ".tools"
        $NodeDir = Join-Path $Tools "node-$NodeVer-win-x64"
        if (-not (Test-Path (Join-Path $NodeDir "npm.cmd"))) {
            Write-Host "Downloading portable Node.js $NodeVer for the dashboard build..."
            New-Item -ItemType Directory -Force $Tools | Out-Null
            $Zip = Join-Path $Tools "node.zip"
            $ProgressPreference = "SilentlyContinue"
            Invoke-WebRequest "https://nodejs.org/dist/$NodeVer/node-$NodeVer-win-x64.zip" -OutFile $Zip -UseBasicParsing
            if ((Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower() -ne $NodeSha) {
                Remove-Item $Zip
                throw "Node.js download failed its checksum; not using it."
            }
            Expand-Archive $Zip -DestinationPath $Tools -Force
            Remove-Item $Zip
        }
        $env:Path = "$NodeDir;$env:Path"
    }
    Write-Host "Building the dashboard..."
    Push-Location dashboard
    try {
        Invoke-Checked "npm ci" { npm.cmd ci --no-audit --no-fund }
        Invoke-Checked "dashboard build" { npm.cmd run build }
    } finally { Pop-Location }
}

if ($SetupOnly) { Write-Host "Setup complete."; exit 0 }

# 4. Optional: start automatically at log-on (a launcher in the user's Startup folder; no admin rights)
if ($InstallAutostart) {
    $line = "start `"btc5m paper test`" /min powershell.exe -ExecutionPolicy Bypass -NoProfile -File `"$PSCommandPath`" -Port $Port"
    Set-Content -Path $StartupFile -Value "@echo off`r`n$line`r`n" -Encoding ASCII
    Write-Host "Autostart installed: $StartupFile"
}

# 5. Keep Windows awake while this script runs (released automatically when it stops)
Add-Type -Namespace Btc5m -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Btc5m.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

# 6. Run the server; restart it if it ever stops
New-Item -ItemType Directory -Force "data\logs" | Out-Null
$env:BTC5M_PERSIST = "1"
$env:PYTHONUTF8 = "1"
Write-Host ""
Write-Host "btc5m paper test is running. Dashboard: http://127.0.0.1:$Port"
Write-Host "Logs: $Root\data\logs\serve.log"
Write-Host "Leave this window open (closing it stops the test)."
$ErrorActionPreference = "Continue"
while ($true) {
    # Relative paths without spaces, so cmd.exe needs no inner quoting
    cmd.exe /c ".venv\Scripts\btc5m.exe serve --host 127.0.0.1 --port $Port >> data\logs\serve.log 2>&1"
    $code = $LASTEXITCODE
    if ($code -eq 3) { Write-Host "Another btc5m server is already writing this ledger. Exiting."; exit 3 }
    Write-Host "$(Get-Date -Format s) server stopped (exit $code); restarting in 10 s..."
    Start-Sleep -Seconds 10
}
