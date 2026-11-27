param(
    [string]$RuntimeSeed,
    [switch]$CompileOnly,
    [switch]$KmCalibration,
    [switch]$AutoTracking,
    [ValidatePattern('^[a-zA-Z0-9.-]+$')][string]$PackageName = 'VisionStudio-V1.1'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$app = Join-Path $root 'apps/vision-studio'
$build = Join-Path $root 'build'
$target = Join-Path $build 'vision-studio-target'
$toolchain = Join-Path $build 'vision-studio-toolchain'
if (Test-Path -LiteralPath (Join-Path $toolchain 'rustup')) {
    $msvcRoot = Join-Path $toolchain 'msvc-tools/Contents/VC/Tools/MSVC'
    $msvc = Get-ChildItem -LiteralPath $msvcRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1
    $msvcBin = Join-Path $msvc.FullName 'bin/Hostx64/x64'
    $sdk = Join-Path $toolchain 'xwin-sdk'
    $resources = Get-ChildItem -LiteralPath (Join-Path $toolchain 'sdk-build-tools/bin') -Directory |
        Sort-Object Name -Descending | Select-Object -First 1
    $env:RUSTUP_HOME = Join-Path $toolchain 'rustup'
    $env:CARGO_HOME = Join-Path $toolchain 'cargo'
    $env:PATH = @((Join-Path $env:CARGO_HOME 'bin'), $msvcBin, (Join-Path $resources.FullName 'x64'), $env:PATH) -join ';'
    $env:INCLUDE = (@('crt/include','sdk/include/shared','sdk/include/ucrt','sdk/include/um','sdk/include/winrt') |
        ForEach-Object { Join-Path $sdk $_ }) -join ';'
    $env:LIB = (@('crt/lib/x64','sdk/lib/um/x64','sdk/lib/ucrt/x64') | ForEach-Object { Join-Path $sdk $_ }) -join ';'
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = Join-Path $msvcBin 'link.exe'
}
$env:CARGO_TARGET_DIR = $target
Push-Location $app
try {
    npm exec -- vue-tsc --noEmit
    if ($LASTEXITCODE -ne 0) { throw 'Vue type check failed' }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'Vue production build failed' }
    cargo build --manifest-path src-tauri/Cargo.toml --release --features custom-protocol --target x86_64-pc-windows-msvc
    if ($LASTEXITCODE -ne 0) { throw 'Tauri release build failed' }
} finally {
    Pop-Location
}
$exe = Join-Path $target 'x86_64-pc-windows-msvc/release/vision-studio.exe'
if ($CompileOnly) { Write-Output $exe; return }
if (!$RuntimeSeed) { $RuntimeSeed = Join-Path $build 'trt-portable' }
$seed = (Resolve-Path -LiteralPath $RuntimeSeed).Path
if (!(Test-Path -LiteralPath (Join-Path $seed 'python313._pth'))) { throw 'Expected portable Python 3.13 runtime' }
$stamp = (Get-Date -Format yyyyMMdd-HHmmss) + '-' + [Guid]::NewGuid().ToString('N').Substring(0,6)
$stage = (New-Item -ItemType Directory -Path (Join-Path $build "VisionStudio-V1.1/$stamp")).FullName
$runtime = (New-Item -ItemType Directory -Path (Join-Path $stage 'vision-runtime')).FullName
Get-ChildItem -LiteralPath $seed -File | Copy-Item -Destination $runtime
Copy-Item -LiteralPath (Join-Path $seed 'site-packages') -Destination $runtime -Recurse
$vision = Join-Path $root 'apps/vision-phase2'
foreach ($name in 'worker.py','vision_model.py','ndi_capture.py','cuda_check.py','trt_engine.py') {
    Copy-Item -LiteralPath (Join-Path $vision $name) -Destination $runtime
}
& (Join-Path $PSScriptRoot 'build-vision-fast-pixels.ps1') -OutputDirectory $runtime
$model = $env:VISION_ONNX_MODEL
if (!$model) { $model = Join-Path $env:USERPROFILE 'Downloads/1013v26s_640.onnx' }
Copy-Item -LiteralPath $model -Destination (Join-Path $stage '1013v26s_640.onnx')
Copy-Item -LiteralPath $exe -Destination (Join-Path $stage 'VisionStudio.exe')
Copy-Item -LiteralPath (Join-Path $app 'README.md') -Destination $stage
foreach ($name in 'Start-Vision-CUDA.cmd','README-CUDA.txt','setup-vision-cuda.py','cuda-packages.json') {
    Copy-Item -LiteralPath (Join-Path $root "apps/cuda-runtime-setup/$name") -Destination $stage
}
foreach ($name in 'Install-TensorRT.cmd','README-TensorRT.txt','setup-vision-tensorrt.py','tensorrt-packages.json') {
    Copy-Item -LiteralPath (Join-Path $root "apps/tensorrt-runtime-setup/$name") -Destination $stage
}
$compiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
$monitor = Join-Path $stage 'VisionKmMonitor.exe'
$compilerArgs = @('/nologo','/platform:x64','/target:exe','/optimize+','/warnaserror+','/codepage:65001',
    $(if ($AutoTracking) { '/define:KMBOX_AUTO_TRACKING' } else { '/define:KMBOX_READONLY' }),'/r:System.Web.Extensions.dll',"/out:$monitor",
    (Join-Path $app 'sidecar/Monitor.cs'), (Join-Path $app 'sidecar/ButtonConfirmation.cs'),
    (Join-Path $root 'apps/hardware-check/KmBox.cs'))
& $compiler @compilerArgs
if ($LASTEXITCODE -ne 0) { throw 'KMBox monitor build failed' }
$binaries = @((Join-Path $stage 'VisionStudio.exe'), $monitor)
$confirmation = Join-Path $stage 'VisionButtonStatus.exe'
$confirmationArgs = @('/nologo','/platform:x64','/target:winexe','/optimize+','/warnaserror+','/codepage:65001',
    '/r:System.Windows.Forms.dll','/r:System.Drawing.dll',"/out:$confirmation",
    (Join-Path $app 'sidecar/ButtonStatusApp.cs'), (Join-Path $app 'sidecar/ButtonConfirmation.cs'))
& $compiler @confirmationArgs
if ($LASTEXITCODE -ne 0) { throw 'Device A button confirmation build failed' }
$binaries += $confirmation
if ($KmCalibration) {
    $calibration = Join-Path $stage 'VisionKmCalibration.exe'
    $hardware = Join-Path $root 'apps/hardware-check'
    $calibrationArgs = @('/nologo','/platform:x64','/target:winexe','/optimize+','/warnaserror+','/codepage:65001',
        '/define:KMBOX_CALIBRATION','/r:System.Windows.Forms.dll','/r:System.Drawing.dll',"/out:$calibration",
        (Join-Path $hardware 'Program.cs'),(Join-Path $hardware 'KmBox.cs'),(Join-Path $hardware 'KmBoxSelfTest.cs'))
    & $compiler @calibrationArgs
    if ($LASTEXITCODE -ne 0) { throw 'KMBox calibration build failed' }
    $binaries += $calibration
}
foreach ($binary in $binaries) {
    $bytes = [IO.File]::ReadAllBytes($binary)
    if ([BitConverter]::ToUInt16($bytes, [BitConverter]::ToInt32($bytes,60)+4) -ne 0x8664) { throw "Not an x64 executable: $binary" }
}
$dist = (New-Item -Force -ItemType Directory -Path (Join-Path $root 'dist')).FullName
$zip = Join-Path $dist "$PackageName-windows-x64.zip"
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -Force
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath ($zip + '.sha256') -Encoding ascii
Write-Output $zip
Write-Output "SHA256 $hash"
