param([Parameter(Mandatory=$true)][string]$OutputDirectory)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$scratch = (New-Item -ItemType Directory -Path (Join-Path $root ("build/fast-pixels-" + [Guid]::NewGuid().ToString('N').Substring(0,8)))).FullName
$output = Join-Path ((Resolve-Path -LiteralPath $OutputDirectory).Path) 'VisionFastPixels.dll'
$object = Join-Path $scratch 'pixels.obj'
$library = Join-Path $scratch 'pixels.lib'
$compileFlags = @('/nologo','/O2','/MT','/LD','/W4','/WX',"/Fo:$object","/Fe:$output",
    (Join-Path $root 'apps/vision-phase2/fast_pixels.c'),'/link',"/IMPLIB:$library")
& cl.exe @compileFlags
if ($LASTEXITCODE -ne 0) { throw 'VisionFastPixels Windows x64 build failed' }
