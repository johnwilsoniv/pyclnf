<#
.SYNOPSIS
    Build the cpp_warp Python extension on Windows.
.DESCRIPTION
    Builds cpp_warp.cp<PY>-win_amd64.pyd against a pre-built OpenCV 4.12 install.
    Mirrors the macOS build.sh but uses VS 2022 BuildTools and an OPENCV_DIR env
    var instead of the Homebrew path.

    Prerequisites:
      - VS 2022 BuildTools with C++ Desktop workload (provides cl.exe, MSBuild)
      - Python 3.10/3.11/3.12 with pip
      - pip-installed pybind11 and cmake (so the venv's cmake is on PATH)
      - OpenCV 4.12.0 Windows pre-built unpacked somewhere, with OPENCV_DIR env
        var pointing at its install root (the dir containing build/OpenCVConfig.cmake)
.EXAMPLE
    $env:OPENCV_DIR = "C:\Users\User\Documents\opencv"
    cd pyclnf\cpp_warp
    .\build.ps1
#>
[CmdletBinding()]
param(
    [string]$PythonExe = (Get-Command python).Source,
    [string]$CMakeGenerator = "Visual Studio 17 2022",
    [string]$Architecture = "x64"
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
Set-Location $scriptDir

Write-Output "Building cpp_warp extension..."
Write-Output "================================"
Write-Output "  python      = $PythonExe"
Write-Output "  generator   = $CMakeGenerator"
Write-Output "  arch        = $Architecture"
Write-Output "  OPENCV_DIR  = $env:OPENCV_DIR"
Write-Output ""

if (-not $env:OPENCV_DIR) {
    throw "OPENCV_DIR env var is not set. Point it at the OpenCV 4.12 install root."
}
if (-not (Test-Path "$env:OPENCV_DIR\build\OpenCVConfig.cmake")) {
    throw "OPENCV_DIR=$env:OPENCV_DIR does not contain build\OpenCVConfig.cmake."
}

$cmakeExe = & $PythonExe -c "import cmake, os; print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin', 'cmake.exe'))"
if (-not (Test-Path $cmakeExe)) {
    throw "cmake binary not found at $cmakeExe. Install via: $PythonExe -m pip install cmake"
}

$buildDir = Join-Path $scriptDir "build"
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
New-Item -ItemType Directory -Force $buildDir | Out-Null
Set-Location $buildDir

& $cmakeExe .. `
    -G "$CMakeGenerator" -A $Architecture `
    "-DPython3_EXECUTABLE=$PythonExe" `
    -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (exit $LASTEXITCODE)" }

& $cmakeExe --build . --config Release
if ($LASTEXITCODE -ne 0) { throw "cmake build failed (exit $LASTEXITCODE)" }

# Find the built .pyd. CMakeLists.txt sets LIBRARY_OUTPUT_DIRECTORY=CMAKE_SOURCE_DIR
# but on Windows MSBuild places the runtime artifact (.dll/.pyd) in
# CMAKE_SOURCE_DIR\Release\ (multi-config generator behavior). Search both
# locations so the script is robust to either layout.
$pyd = Get-ChildItem -Path "$scriptDir\Release", "$buildDir\Release", $buildDir `
    -Filter "cpp_warp*.pyd" -Recurse -ErrorAction SilentlyContinue `
    | Select-Object -First 1
if (-not $pyd) {
    throw "Did not find cpp_warp*.pyd under $scriptDir\Release\ or $buildDir\Release\"
}
Copy-Item $pyd.FullName (Join-Path $scriptDir $pyd.Name) -Force
Write-Output ""
Write-Output "================================"
Write-Output "Build complete: $($pyd.Name)"
Write-Output "================================"

# Also copy the OpenCV runtime DLL the .pyd will need (otherwise import fails
# with LoadLibrary error). The world DLL has the version suffixed.
$opencvBin = "$env:OPENCV_DIR\build\x64\vc16\bin"
if (Test-Path $opencvBin) {
    $worldDll = Get-ChildItem "$opencvBin\opencv_world*.dll" | Where-Object { $_.Name -notmatch "d\.dll$" } | Select-Object -First 1
    if ($worldDll) {
        Copy-Item $worldDll.FullName (Join-Path $scriptDir $worldDll.Name) -Force
        Write-Output "Bundled OpenCV runtime: $($worldDll.Name)"
    }
}

Set-Location $scriptDir
Write-Output ""
Write-Output "Testing import..."
& $PythonExe -c "import sys, os; sys.path.insert(0, os.getcwd()); import cpp_warp; print(f'cpp_warp loaded: BACKEND inferred (file-loaded); OpenCV version: {cpp_warp.get_opencv_version()}')"
