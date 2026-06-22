param(
    [string]$SourceDir = "C:\Users\Arik\Documents\uni\ResearchMethods\external\Metric-FF\Metric-FF-v2.1",
    [switch]$CheckOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Find-Tool {
    param([string[]]$Names)
    foreach ($name in $Names) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $cmd) {
            return $cmd.Source
        }
    }
    return $null
}

function Write-Status {
    param([string]$Label, [string]$Value)
    Write-Host ("{0,-12}: {1}" -f $Label, $Value)
}

if (-not (Test-Path -Path $SourceDir)) {
    throw "Metric-FF source directory not found: $SourceDir"
}

$makeTool = Find-Tool -Names @("mingw32-make", "make")
$gccTool = Find-Tool -Names @("gcc")
$bisonTool = Find-Tool -Names @("win_bison", "bison")
$flexTool = Find-Tool -Names @("win_flex", "flex")

Write-Status -Label "SourceDir" -Value $SourceDir
Write-Status -Label "make" -Value ($(if ($makeTool) { $makeTool } else { "missing" }))
Write-Status -Label "gcc" -Value ($(if ($gccTool) { $gccTool } else { "missing" }))
Write-Status -Label "bison" -Value ($(if ($bisonTool) { $bisonTool } else { "missing" }))
Write-Status -Label "flex" -Value ($(if ($flexTool) { $flexTool } else { "missing" }))

$binaryCandidates = @(
    (Join-Path $SourceDir "ff.exe"),
    (Join-Path $SourceDir "ff")
)

$existingBinary = $binaryCandidates | Where-Object { Test-Path -Path $_ } | Select-Object -First 1
if ($existingBinary -and -not $Force) {
    Write-Status -Label "binary" -Value "exists ($existingBinary)"
    if ($CheckOnly) {
        exit 0
    }

    Write-Host "Binary already exists. Use -Force to rebuild."
    exit 0
}

if ($CheckOnly) {
    if ($makeTool -and $gccTool -and $bisonTool -and $flexTool) {
        Write-Host "Toolchain check passed. You can build Metric-FF."
        exit 0
    }

    Write-Host "Toolchain check failed. Install a MinGW/MSYS2 toolchain with make, gcc, bison, and flex."
    exit 1
}

if (-not ($makeTool -and $gccTool -and $bisonTool -and $flexTool)) {
    throw "Missing required build tools. Run with -CheckOnly to inspect tool availability."
}

Push-Location $SourceDir
try {
    & $makeTool clean | Out-Host
    & $makeTool ff | Out-Host
}
finally {
    Pop-Location
}

$builtBinary = $binaryCandidates | Where-Object { Test-Path -Path $_ } | Select-Object -First 1
if (-not $builtBinary) {
    throw "Metric-FF build completed but no ff binary was found in $SourceDir"
}

Write-Status -Label "binary" -Value "built ($builtBinary)"
Write-Host "Done."

