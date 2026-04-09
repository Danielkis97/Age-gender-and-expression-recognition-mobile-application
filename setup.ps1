# One-time / fresh machine setup: create venv and install dependencies from requirements.txt.
# TensorFlow needs Python 3.10–3.12 on Windows (no wheels for 3.13+ / 3.14 yet).
# Run from project root:  .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
$pythonExe = Get-Command python -ErrorAction SilentlyContinue
$chosen = $null
$useDirectPython = $false
$pythonCmd = $null

if ($pyLauncher) {
    foreach ($flag in @("-3.12", "-3.11", "-3.10")) {
        $verOut = & py $flag --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $chosen = $flag
            Write-Host "Using $verOut  (py $flag)"
            break
        }
    }
}

if (-not $chosen) {
    $pythonCandidates = @()
    if ($pythonExe) {
        $pythonCandidates += "python"
    }
    $pythonCandidates += @(
        "$env:LocalAppData\Programs\Python\Python312\python.exe",
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:LocalAppData\Programs\Python\Python310\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe",
        "C:\Program Files\Python310\python.exe"
    )

    foreach ($candidate in ($pythonCandidates | Select-Object -Unique)) {
        if ($candidate -ne "python" -and -not (Test-Path $candidate)) {
            continue
        }
        $verOut = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $verOut -match "Python 3\.(10|11|12)\.") {
            $useDirectPython = $true
            $pythonCmd = $candidate
            Write-Host "Using $verOut  ($candidate)"
            break
        }
    }
}

if (-not $chosen -and -not $useDirectPython) {
    Write-Host ""
    Write-Host "No compatible Python 3.10–3.12 found. TensorFlow cannot install on Python 3.13+ yet (pip reports 'No matching distribution')." -ForegroundColor Yellow
    Write-Host "Fix: install Python 3.12 (64-bit) from https://www.python.org/downloads/ and enable 'Add python.exe to PATH'." -ForegroundColor Yellow
    Write-Host "If available, also enable the 'py launcher' option." -ForegroundColor Yellow
    Write-Host "Then run one of these:" -ForegroundColor Yellow
    Write-Host "  py -3.12 -m venv .venv" -ForegroundColor Yellow
    Write-Host "  or" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

if (Test-Path ".venv") {
    Write-Host "Removing existing .venv ..."
    try {
        Remove-Item -Recurse -Force .venv -ErrorAction Stop
    } catch {
        Write-Host ""
        Write-Host "Cannot delete .venv (files in use). Do this:" -ForegroundColor Yellow
        Write-Host "  1) Run: deactivate   (in every shell until prompt has no (.venv))" -ForegroundColor Yellow
        Write-Host "  2) Close PyCharm / other terminals using this project's .venv" -ForegroundColor Yellow
        Write-Host "  3) Delete the .venv folder in Explorer, or run Remove-Item again in a fresh PowerShell" -ForegroundColor Yellow
        exit 1
    }
}

if ($useDirectPython) {
    Write-Host "Creating virtual environment with $pythonCmd ..."
    & $pythonCmd -m venv .venv
} else {
    Write-Host "Creating virtual environment with py $chosen ..."
    & py $chosen -m venv .venv
}

Write-Host "Activating .venv ..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing requirements ..."
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Done. Activate later with .\.venv\Scripts\Activate.ps1"
