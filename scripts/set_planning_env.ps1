param(
    [string]$WorkspaceRoot = "C:\Users\Arik\Documents\uni\ResearchMethods",
    [string]$MetricFFDirectory,
    [string]$EnhspJarPath,
    [string]$ValidatorDirectory,
    [string]$ConvexHullErrorPath,
    [switch]$Persist
)

$ErrorActionPreference = "Stop"

if (-not $MetricFFDirectory) {
    $MetricFFDirectory = Join-Path $WorkspaceRoot "external\Metric-FF\Metric-FF-v2.1"
}
if (-not $EnhspJarPath) {
    $EnhspJarPath = "C:\tools\enhsp\enhsp.jar"
}
if (-not $ValidatorDirectory) {
    $ValidatorDirectory = "C:\tools\VAL"
}
if (-not $ConvexHullErrorPath) {
    $ConvexHullErrorPath = Join-Path $WorkspaceRoot "logs\convex_hull_error.txt"
}

$convexParent = Split-Path -Parent $ConvexHullErrorPath
if ($convexParent -and -not (Test-Path -Path $convexParent)) {
    New-Item -ItemType Directory -Path $convexParent -Force | Out-Null
}
if (-not (Test-Path -Path $ConvexHullErrorPath)) {
    New-Item -ItemType File -Path $ConvexHullErrorPath -Force | Out-Null
}

$vars = @{
    METRIC_FF_DIRECTORY = $MetricFFDirectory
    ENHSP_FILE_PATH = $EnhspJarPath
    CONVEX_HULL_ERROR_PATH = $ConvexHullErrorPath
    VALIDATOR_DIRECTORY = $ValidatorDirectory
}

foreach ($key in $vars.Keys) {
    $value = $vars[$key]
    Set-Item -Path ("Env:{0}" -f $key) -Value $value
    if ($Persist) {
        [System.Environment]::SetEnvironmentVariable($key, $value, "User")
    }
}

Write-Host "Configured planning environment variables:"
foreach ($key in $vars.Keys) {
    Write-Host ("  {0}={1}" -f $key, $vars[$key])
}

if ($Persist) {
    Write-Host "Variables were persisted at User scope. Restart terminals to load them automatically."
} else {
    Write-Host "Variables are set for this PowerShell session only."
}


