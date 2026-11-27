param(
    [ValidatePattern('^[a-zA-Z0-9.-]+$')][string]$PackageName = 'VisionStudio-V1.1-Alpha'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$stage = (New-Item -ItemType Directory -Path (Join-Path $root ('build/vision-source-' + [Guid]::NewGuid().ToString('N').Substring(0,8)))).FullName
$entries = @(
    'apps/vision-studio/src', 'apps/vision-studio/sidecar', 'apps/vision-studio/index.html',
    'apps/vision-studio/vite.config.js', 'apps/vision-studio/tsconfig.json',
    'apps/vision-studio/package.json', 'apps/vision-studio/package-lock.json', 'apps/vision-studio/README.md',
    'apps/vision-studio/src-tauri/src', 'apps/vision-studio/src-tauri/icons', 'apps/vision-studio/src-tauri/capabilities',
    'apps/vision-studio/src-tauri/build.rs', 'apps/vision-studio/src-tauri/Cargo.toml',
    'apps/vision-studio/src-tauri/Cargo.lock', 'apps/vision-studio/src-tauri/tauri.conf.json',
    'apps/cuda-runtime-setup', 'apps/tensorrt-runtime-setup', 'scripts', '.gitignore'
)
foreach ($folder in 'apps/vision-phase2','apps/hardware-check') {
    $entries += Get-ChildItem -LiteralPath (Join-Path $root $folder) -File |
        ForEach-Object { "$folder/$($_.Name)" }
}
foreach ($entry in $entries) {
    $target = Join-Path $stage $entry
    New-Item -Force -ItemType Directory (Split-Path $target -Parent) | Out-Null
    Copy-Item -LiteralPath (Join-Path $root $entry) -Destination $target -Recurse
}
Copy-Item -LiteralPath (Join-Path $root 'apps/vision-studio/README.md') -Destination $stage
$dist = (New-Item -ItemType Directory -Force (Join-Path $root 'dist')).FullName
$zip = Join-Path $dist "$PackageName-Source.zip"
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -Force
$sha = (Get-FileHash -LiteralPath $zip).Hash.ToLowerInvariant()
"$sha  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath ($zip + '.sha256') -Encoding ascii
Write-Output $zip
