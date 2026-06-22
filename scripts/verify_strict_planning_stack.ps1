param(
    [string]$WorkspaceRoot = "C:\Users\Arik\Documents\uni\ResearchMethods",
    [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"

$checks = @(
    @{ Label = "WSL command"; Path = "wsl.exe"; IsCommand = $true },
    @{ Label = "Metric-FF source"; Path = (Join-Path $WorkspaceRoot "external\Metric-FF\Metric-FF-v2.1"); IsCommand = $false },
    @{ Label = "Metric-FF Windows binary"; Path = (Join-Path $WorkspaceRoot "external\Metric-FF\Metric-FF-v2.1\ff.exe"); IsCommand = $false },
    @{ Label = "Metric-FF POSIX binary"; Path = (Join-Path $WorkspaceRoot "external\Metric-FF\Metric-FF-v2.1\ff"); IsCommand = $false },
    @{ Label = "VAL source"; Path = (Join-Path $WorkspaceRoot "external\VAL"); IsCommand = $false },
    @{ Label = "VAL build dir"; Path = (Join-Path $WorkspaceRoot "external\VAL-build"); IsCommand = $false }
)

foreach ($check in $checks) {
    if ($check.IsCommand) {
        $cmd = Get-Command $check.Path -ErrorAction SilentlyContinue
        Write-Host ("{0,-24}: {1}" -f $check.Label, $(if ($cmd) { $cmd.Source } else { "missing" }))
    } else {
        Write-Host ("{0,-24}: {1}" -f $check.Label, (Test-Path $check.Path))
    }
}

$wslStatus = (& cmd.exe /c "wsl.exe --status 2>&1" | Out-String)
Write-Host "WSL status:"
Write-Host $wslStatus

if ($wslStatus -match "not installed") {
    Write-Host "Installed distros        : none (WSL not installed)"
    exit 0
}

$distroList = & cmd.exe /c "wsl.exe --list --quiet 2>&1" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
Write-Host ("Installed distros        : {0}" -f ($(if ($distroList) { ($distroList -join ', ') } else { 'none' })))

if ($distroList -contains $Distro) {
    $normalizedWindowsPath = $WorkspaceRoot -replace "\\", "/"
    if ($normalizedWindowsPath -notmatch "^([A-Za-z]):/(.+)$") {
        throw "WorkspaceRoot must be an absolute Windows path like C:/... Got: $WorkspaceRoot"
    }
    $workspaceWslPath = "/mnt/{0}/{1}" -f $matches[1].ToLower(), $matches[2]
    $verifyScriptLines = @(
        "set -e",
        "set -u",
        "set -o pipefail",
        "printf 'java                    : '",
        "command -v java || true",
        "printf 'gcc                     : '",
        "command -v gcc || true",
        "printf 'make                    : '",
        "command -v make || true",
        "printf 'bison                   : '",
        "command -v bison || true",
        "printf 'flex                    : '",
        "command -v flex || true",
        "printf 'metric-ff               : '",
        "if [ -x '$workspaceWslPath/external/Metric-FF/Metric-FF-v2.1/ff' ]; then echo present; else echo missing; fi",
        "printf 'val-validate            : '",
        "if [ -x '$workspaceWslPath/external/VAL-build/Validate' ]; then echo present; else echo missing; fi"
    )
    $verifyScriptUnix = ($verifyScriptLines -join "`n") -replace "`r", ""
    $verifyScriptUnix | & wsl.exe -d $Distro -- bash -lc "tr -d '\r' | bash -s"
}


