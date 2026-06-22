param(
    [string]$Distro = "Ubuntu",
    [string]$WorkspaceWindowsPath = "C:\Users\Arik\Documents\uni\ResearchMethods"
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "Required command not found: $Name"
    }
}

Require-Command -Name "wsl.exe"

$wslStatus = & wsl.exe --status 2>&1 | Out-String
if ($wslStatus -match "not installed") {
    throw "WSL is not installed. Run `wsl.exe --install` in an elevated PowerShell window, reboot, then rerun this script."
}

$distroList = & wsl.exe --list --quiet 2>&1 | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($distroList -notcontains $Distro) {
    throw "WSL distro '$Distro' not found. Install it first (for example: `wsl.exe --install -d $Distro`)."
}

$normalizedWindowsPath = $WorkspaceWindowsPath -replace "\\", "/"
if ($normalizedWindowsPath -notmatch "^([A-Za-z]):/(.+)$") {
    throw "Workspace path must be an absolute Windows path like C:/... Got: $WorkspaceWindowsPath"
}
$workspaceWslPath = "/mnt/{0}/{1}" -f $matches[1].ToLower(), $matches[2]

$workspaceExists = (& wsl.exe -d $Distro -- bash -lc "[ -d '$workspaceWslPath' ] && echo yes || echo no" 2>&1 | Out-String).Trim()
if ($workspaceExists -ne "yes") {
    throw "Workspace path is not accessible in WSL: $workspaceWslPath"
}

$setupScriptLines = @(
    "set -e",
    "set -u",
    "set -o pipefail",
    "export DEBIAN_FRONTEND=noninteractive",
    "sudo apt-get update",
    "sudo apt-get install -y build-essential g++ gcc make bison flex default-jre git cmake python3 python3-pip",
    "cd '$workspaceWslPath/external/Metric-FF/Metric-FF-v2.1'",
    "make clean || true",
    "make ff",
    "if [ ! -x ./ff ]; then",
    "  echo 'Metric-FF build failed: ./ff not found' >&2",
    "  exit 1",
    "fi",
    "mkdir -p '$workspaceWslPath/external/VAL-build'",
    "if [ ! -d '$workspaceWslPath/external/VAL/.git' ]; then",
    "  git clone https://github.com/KCL-Planning/VAL '$workspaceWslPath/external/VAL'",
    "fi",
    "cd '$workspaceWslPath/external/VAL'",
    "cmake -S . -B '$workspaceWslPath/external/VAL-build'",
    "cmake --build '$workspaceWslPath/external/VAL-build' -j",
    "if [ ! -x '$workspaceWslPath/external/VAL-build/Validate' ]; then",
    "  echo 'VAL build failed: Validate binary not found' >&2",
    "  exit 1",
    "fi",
    "echo 'WSL planning environment setup completed successfully.'"
)

$setupScriptUnix = ($setupScriptLines -join "`n") -replace "`r", ""
$setupScriptUnix | & wsl.exe -d $Distro -- bash -lc "tr -d '\r' | bash -s"

