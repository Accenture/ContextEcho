[CmdletBinding()]
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $DonateArgs
)

$ErrorActionPreference = "Stop"

$Spec = if ($env:CONTEXTECHO_DONATE_SPEC) { $env:CONTEXTECHO_DONATE_SPEC } else { "git+https://github.com/Accenture/ContextEcho.git" }
$env:CONTEXTECHO_RELAY_URL = if ($env:CONTEXTECHO_RELAY_URL) { $env:CONTEXTECHO_RELAY_URL } else { "https://contextecho2026-context-echo-donation-relay.hf.space" }
$DonatePythonCandidates = if ($env:CONTEXTECHO_DONATE_PYTHON) {
  @($env:CONTEXTECHO_DONATE_PYTHON)
} elseif ($env:CONTEXTECHO_DONATE_PYTHONS) {
  $env:CONTEXTECHO_DONATE_PYTHONS -split "\s+"
} else {
  @("3.12", "3.11", "3.13", "3.10")
}

function Invoke-CandidatePython {
  param(
    [string[]] $Candidate,
    [string[]] $Arguments
  )
  $exe = $Candidate[0]
  $baseArgs = @()
  if ($Candidate.Count -gt 1) {
    $baseArgs = $Candidate[1..($Candidate.Count - 1)]
  }
  & $exe @baseArgs @Arguments
}

function Find-Python {
  $candidates = @(
    @("py", "-3.12"),
    @("py", "-3.11"),
    @("py", "-3.13"),
    @("py", "-3.10"),
    @("py", "-3"),
    @("python"),
    @("python3")
  )
  foreach ($candidate in $candidates) {
    if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) {
      continue
    }
    try {
      Invoke-CandidatePython -Candidate $candidate -Arguments @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)") *> $null
      if ($LASTEXITCODE -eq 0) {
        return $candidate
      }
    } catch {
      continue
    }
  }
  return $null
}

function Explain-Failure {
  param([int] $ExitCode)
  Write-Error @"
ContextEcho could not start the local donation wizard.

What to try next:
  1. Check that this machine can reach GitHub and PyPI.
  2. Install Python 3.10-3.13 from https://www.python.org/downloads/windows/ or ask IT to allow one of: $($DonatePythonCandidates -join ", ").
  3. Rerun the same command; ContextEcho reuses its private cache.

Debug details:
  OS: Windows
  wizard python candidates: $($DonatePythonCandidates -join ", ")
  cache: $CacheRoot
  exit code: $ExitCode
"@
}

$PythonCmd = Find-Python
if (-not $PythonCmd) {
  Write-Error "ContextEcho needs Python 3.8+ to bootstrap the local wizard. Install Python 3, then rerun this command."
  exit 1
}

$cacheBase = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME ".cache" }
$CacheRoot = Join-Path $cacheBase "contextecho-donate"
$UvVenv = Join-Path $CacheRoot "uv-venv"
$env:UV_CACHE_DIR = if ($env:UV_CACHE_DIR) { $env:UV_CACHE_DIR } else { Join-Path $CacheRoot "uv-cache" }
$env:UV_PYTHON_INSTALL_DIR = if ($env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR } else { Join-Path $CacheRoot "uv-python" }

New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null

$UvCmd = $null
if (Get-Command uv -ErrorAction SilentlyContinue) {
  $UvCmd = "uv"
} else {
  Write-Host "[ContextEcho] uv not found; creating a private bootstrap environment..."
  try {
    Invoke-CandidatePython -Candidate $PythonCmd -Arguments @("-m", "venv", $UvVenv)
    if ($LASTEXITCODE -ne 0) {
      Explain-Failure $LASTEXITCODE
      exit $LASTEXITCODE
    }
    $venvPython = Join-Path $UvVenv "Scripts\python.exe"
    & $venvPython -m pip install --upgrade pip uv
    if ($LASTEXITCODE -ne 0) {
      Explain-Failure $LASTEXITCODE
      exit $LASTEXITCODE
    }
    $UvCmd = Join-Path $UvVenv "Scripts\uv.exe"
  } catch {
    Explain-Failure 1
    exit 1
  }
}

Write-Host "[ContextEcho] starting local donation wizard..."
Write-Host "[ContextEcho] raw sessions stay on this machine; the browser wizard will open automatically."

$lastRc = 1
foreach ($pyVersion in $DonatePythonCandidates) {
  Write-Host "[ContextEcho] trying Python $pyVersion for the local wizard..."
  $uvArgs = @("run", "--refresh", "--no-project", "--python", $pyVersion, "--with", $Spec, "contextecho-donate")
  if ($DonateArgs) {
    $uvArgs += $DonateArgs
  }
  try {
    & $UvCmd @uvArgs
    $rc = $LASTEXITCODE
    if ($rc -eq 0) {
      exit 0
    }
    $lastRc = $rc
    Write-Warning "[ContextEcho] Python $pyVersion did not start the wizard; trying the next supported runtime if available."
  } catch {
    $lastRc = 1
    Write-Warning "[ContextEcho] Python $pyVersion did not start the wizard; trying the next supported runtime if available."
  }
}

Explain-Failure $lastRc
exit $lastRc
