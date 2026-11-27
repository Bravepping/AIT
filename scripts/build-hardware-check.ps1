param(
    [ValidateSet('01','02','02.1','02.2','03')][string]$Version = '01',
    [string]$RuntimeSeed,
    [switch]$SkipGuiCheck
)
$ErrorActionPreference = 'Stop'
if ($Version -eq '02') { $Version = '02.1' }
$root = Split-Path $PSScriptRoot -Parent
$project = Join-Path $root 'apps/hardware-check'
$stamp = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,6)
$build = Join-Path $root "build/HardwareCheck-$Version/$stamp"
$dist = Join-Path $root 'dist'
$package = Join-Path $build 'package'
New-Item -Force -ItemType Directory $build,$dist,$package | Out-Null
$compiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
$exe = Join-Path $package 'VisionHardwareCheck.exe'
$buildLog = Join-Path $build 'build.log'
$testLog = Join-Path $package 'TEST-RESULTS.txt'
[string[]]$sources = Get-ChildItem $project -Filter '*.cs' | ForEach-Object FullName
$compilerArgs = @('/nologo','/target:winexe','/platform:x64','/optimize+','/warnaserror+','/codepage:65001',
    '/r:System.Windows.Forms.dll','/r:System.Drawing.dll',"/out:$exe")
if ($Version -ne '01') {
    $compilerArgs += '/define:PHASE2'
    $compilerArgs += '/r:System.Web.Extensions.dll'
    $phase2 = Join-Path $root 'apps/vision-phase2'
    $sources += Get-ChildItem $phase2 -Filter '*.cs' | ForEach-Object FullName
}
"Windows x64 compile · $(Get-Date -Format o) · $compiler" | Set-Content $buildLog -Encoding UTF8
& $compiler @compilerArgs @sources 2>&1 | Tee-Object -FilePath $buildLog -Append
if ($LASTEXITCODE -ne 0) { throw "Compiler exited $LASTEXITCODE" }
$binary = [IO.File]::ReadAllBytes($exe)
$peOffset = [BitConverter]::ToInt32($binary,60)
if ([BitConverter]::ToUInt16($binary,$peOffset+4) -ne 0x8664) { throw 'Expected an AMD64 executable' }
'BUILD OK · PE AMD64' | Add-Content $buildLog
if (Test-Path -LiteralPath $testLog) { Remove-Item -LiteralPath $testLog }
$test = Start-Process -FilePath $exe -ArgumentList @('--self-test', ('"' + $testLog + '"')) -WindowStyle Hidden -PassThru
if (!$test.WaitForExit(25000)) { $test.Kill(); throw 'KMBox self-test timed out' }
if ($test.ExitCode -ne 0 -or !(Test-Path -LiteralPath $testLog)) {
    if (Test-Path -LiteralPath $testLog) { Get-Content $testLog }
    throw "KMBox self-test exited $($test.ExitCode)"
}
if (!$SkipGuiCheck) {
    $marker = Join-Path $build 'gui-start.txt'
    if (Test-Path -LiteralPath $marker) { Remove-Item -LiteralPath $marker }
    $gui = Start-Process -FilePath $exe -ArgumentList @('--verify-start', ('"' + $marker + '"')) -WindowStyle Hidden -PassThru
    if (!$gui.WaitForExit(15000)) { $gui.Kill(); throw 'GUI startup timed out' }
    if ($gui.ExitCode -ne 0 -or !(Test-Path -LiteralPath $marker) -or (Get-Content -Raw $marker) -ne 'WINDOWS_X64_GUI_STARTED') { throw 'GUI startup failed' }
    'WINDOWS PASS — WinForms startup and clean exit' | Add-Content $testLog -Encoding UTF8
} else {
    'WINDOWS SKIP — GUI startup' | Add-Content $testLog -Encoding UTF8
}
Copy-Item -LiteralPath $buildLog -Destination $package
Copy-Item -LiteralPath (Join-Path $project 'README.txt') -Destination $package
if ($Version -ne '01') {
    & (Join-Path $PSScriptRoot 'stage-vision.ps1') -Package $package -RuntimeSeed $RuntimeSeed -TensorRT:($Version -eq '03')
}
$zip = Join-Path $dist "HardwareCheck-$Version-windows-x64.zip"
Compress-Archive -Path (Join-Path $package '*') -DestinationPath $zip -Force
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
"$hash  $([IO.Path]::GetFileName($zip))" | Set-Content -Encoding ASCII ($zip + '.sha256')
Copy-Item -LiteralPath $testLog -Destination (Join-Path $dist "HardwareCheck-$Version-TEST-RESULTS.txt")
Get-Content -LiteralPath $testLog
Write-Output "$zip"
Write-Output "SHA256 $hash"
