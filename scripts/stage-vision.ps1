param(
    [Parameter(Mandatory=$true)][string]$Package,
    [string]$RuntimeSeed,
    [switch]$TensorRT
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$deps = Join-Path $root 'build/downloads'
$source = Join-Path $root 'apps/vision-phase2'
$runtime = Join-Path $Package 'vision-runtime'
$runtimePython = Join-Path $runtime 'python.exe'
$defaultModel = Join-Path $env:USERPROFILE 'Downloads/1013v26s_640.onnx'
if ($env:VISION_ONNX_MODEL) { $defaultModel = $env:VISION_ONNX_MODEL }
if (!(Test-Path -LiteralPath $defaultModel)) { throw 'Set VISION_ONNX_MODEL to the file 1013v26s_640.onnx' }
$stagedModel = Join-Path $Package '1013v26s_640.onnx'
$testLog = Join-Path $Package 'TEST-RESULTS.txt'
New-Item -Force -ItemType Directory $runtime | Out-Null
if ($RuntimeSeed) {
    $seed = (Resolve-Path -LiteralPath $RuntimeSeed).Path
    $isolated = @(Get-ChildItem -LiteralPath $seed -File -Filter 'python3*._pth')
    if ($isolated.Count -ne 1 -or $isolated[0].BaseName -notin @('python313','python314')) {
        throw 'The offline runtime must use an isolated Windows Python 3.13 or 3.14'
    }
    $abi = $isolated[0].BaseName
    $files = 'python.exe',"$abi.dll",'python3.dll','vcruntime140.dll','vcruntime140_1.dll','LICENSE.txt',
             "$abi.zip","$abi._pth",'site-packages'
    foreach ($item in $files) {
        Copy-Item -LiteralPath (Join-Path $seed $item) -Destination $runtime -Recurse
    }
    if (Test-Path -LiteralPath (Join-Path $seed 'DLLs')) {
        Copy-Item -LiteralPath (Join-Path $seed 'DLLs') -Destination $runtime -Recurse
    }
    foreach ($binary in (Get-ChildItem -LiteralPath $seed -File | Where-Object { $_.Extension -in @('.pyd','.dll') -and $_.Name -notin $files })) {
        Copy-Item -LiteralPath $binary.FullName -Destination $runtime
    }
    & $runtimePython -B -c "import struct, sys; from importlib.metadata import version; assert sys.version_info[:2] in ((3, 13), (3, 14)) and struct.calcsize('P') == 8; assert [version(name) for name in ('numpy', 'pillow', 'onnxruntime-gpu', 'packaging', 'flatbuffers')] == ['2.5.0', '13.0.0', '1.24.4', '26.3', '25.12.19']"
    if ($LASTEXITCODE -ne 0) { throw 'The offline runtime does not match the required Windows x64 Python and package versions' }
} else {
    if ($TensorRT) { throw 'HardwareCheck-03 needs -RuntimeSeed from scripts/stage-tensorrt-runtime.py' }
    New-Item -Force -ItemType Directory $deps | Out-Null
    $requirements = @('numpy==2.5.0','pillow==13.0.0','onnxruntime-gpu==1.24.4','packaging==26.3','flatbuffers==25.12.19')
    & python -m pip download --disable-pip-version-check --no-deps --only-binary=:all: --dest $deps @requirements
    if ($LASTEXITCODE -ne 0) { throw 'Unable to stage the vision dependencies' }
    & python (Join-Path $PSScriptRoot 'stage-python.py') $deps $runtime
    if ($LASTEXITCODE -ne 0) { throw 'Unable to stage Windows Python' }
}
foreach ($component in 'cuda_check.py','ndi_capture.py','vision_model.py','trt_engine.py','worker.py','benchmark_engines.py') {
    Copy-Item -LiteralPath (Join-Path $source $component) -Destination $runtime
}
Copy-Item -LiteralPath $defaultModel -Destination $stagedModel
Copy-Item -LiteralPath (Join-Path $source 'README.txt') -Destination (Join-Path $Package 'README-NDI-ONNX.txt')
$cudaSetup = Join-Path $root 'apps/cuda-runtime-setup'
foreach ($component in 'Start-Vision-CUDA.cmd','README-CUDA.txt','setup-vision-cuda.py','cuda-packages.json') {
    Copy-Item -LiteralPath (Join-Path $cudaSetup $component) -Destination $Package
}
if ($TensorRT) {
    & $runtimePython -B -c "import sys; assert sys.version_info[:2] == (3, 13)"
    if ($LASTEXITCODE -ne 0) { throw 'TensorRT 10.16 requires the Windows x64 Python 3.13 runtime seed' }
    $setup = Join-Path $root 'apps/tensorrt-runtime-setup'
    foreach ($component in 'Install-TensorRT.cmd','setup-vision-tensorrt.py','tensorrt-packages.json','README-TensorRT.txt') {
        Copy-Item -LiteralPath (Join-Path $setup $component) -Destination $Package
    }
}

& $runtimePython -B -X utf8 (Join-Path $source 'test_phase2.py') $stagedModel $runtime |
    Tee-Object -FilePath $testLog -Append
if ($LASTEXITCODE -ne 0) { throw 'Phase 2 integration test failed' }
$marker = Join-Path (Split-Path $Package -Parent) 'vision-gui-start.txt'
if (Test-Path -LiteralPath $marker) { Remove-Item -LiteralPath $marker }
$exe = Join-Path $Package 'VisionHardwareCheck.exe'
$gui = Start-Process -FilePath $exe -ArgumentList @('--verify-vision-start', ('"' + $marker + '"')) -WindowStyle Hidden -PassThru
if (!$gui.WaitForExit(18000)) { $gui.Kill(); throw 'Vision GUI startup timed out' }
if ($gui.ExitCode -ne 0 -or !(Test-Path -LiteralPath $marker) -or (Get-Content -Raw $marker) -ne 'WINDOWS_X64_VISION_SAMPLE_READY') {
    throw 'Vision GUI did not load the model, receive the CUDA report and display the sample result'
}
'WINDOWS PASS — vision GUI loaded the bundled CPU model, received the CUDA report and displayed the sample result' | Add-Content $testLog -Encoding UTF8
Write-Output "Model SHA256 $((Get-FileHash -LiteralPath $stagedModel -Algorithm SHA256).Hash)"
