# =============================================================================
#  build.ps1  --  One-command command-line build (no Visual Studio IDE needed)
# =============================================================================
#
#  Compiles the whole project with nvcc and drops a single executable at
#  .\build\chacha20_demo.exe. It lives alongside the Visual Studio solution so
#  you can build/run from a plain terminal -- handy for quick iteration or CI.
#
#  HOW IT WORKS:
#  nvcc needs the MSVC host compiler (cl.exe) plus that compiler's INCLUDE / LIB
#  environment, which Visual Studio's VsDevCmd.bat configures. The most reliable
#  way to guarantee that on Windows is to run VsDevCmd.bat and nvcc inside the
#  SAME cmd.exe invocation, so nvcc inherits the fully-initialized environment:
#
#      cmd /c "VsDevCmd.bat -arch=amd64 && nvcc ..."
#
#  (VsDevCmd may print a harmless "'vswhere.exe' is not recognized" notice on
#  some VS 2026 installs; it still sets up the environment correctly, and the
#  '&&' only proceeds to nvcc when VsDevCmd returns success.)
#
#  USAGE:
#      .\build.ps1                 # build for this machine's GPU (sm_75 default)
#      .\build.ps1 -Arch sm_86     # target a different compute capability
#      .\build.ps1 -Run            # build, then run the executable
# =============================================================================

param(
    # GPU compute capability to target. The reference card (RTX 2080 SUPER) is
    # Turing = sm_75. Change to match your GPU (e.g. sm_86 Ampere, sm_89 Ada,
    # sm_90 Hopper). Run `nvidia-smi` / check the docs for your card.
    [string]$Arch = "sm_75",

    # If supplied, run the program after a successful build.
    [switch]$Run
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
Set-Location $projectRoot

Write-Host "==> Locating Visual Studio developer environment..." -ForegroundColor Cyan
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Is Visual Studio installed?"
}
$vsPath = & $vswhere -latest -property installationPath
$vsDevCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) {
    throw "VsDevCmd.bat not found under '$vsPath'."
}

# Ensure the output directory exists.
$buildDir = Join-Path $projectRoot "build"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

# The full source list. We compile across folders; -Iinclude lets every file
# find "chacha20.cuh". The .cpp reference is compiled by nvcc as host code.
$sources = "src/main.cu src/chacha20.cu src/chacha20_reference.cpp tests/test_vectors.cu demo/demo.cu"

# nvcc flags:
#   -O3          : optimize host + device aggressively.
#   -std=c++17   : modern C++ for the host portions.
#   -arch=sm_XX  : generate machine code (SASS) for this GPU.
#   -Iinclude    : header search path.
#   NOTE: --use_fast_math is intentionally OMITTED. This is integer crypto; we
#         require exact, bit-reproducible results, never fast-math shortcuts.
$nvccCommand = "nvcc -O3 -std=c++17 -arch=$Arch -Iinclude $sources -o build/chacha20_demo.exe"

Write-Host "==> Compiling for $Arch ..." -ForegroundColor Cyan
# Run VsDevCmd + nvcc in one cmd session. PowerShell expands $vsDevCmd and
# $nvccCommand into this string BEFORE handing it to cmd.exe.
cmd /c "`"$vsDevCmd`" -arch=amd64 -no_logo 1>nul && $nvccCommand"
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE."
}

$exe = Join-Path $buildDir "chacha20_demo.exe"
Write-Host "==> Build succeeded: $exe" -ForegroundColor Green

if ($Run) {
    Write-Host "==> Running...`n" -ForegroundColor Cyan
    & $exe
}
