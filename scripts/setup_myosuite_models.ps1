# Bootstrap the pinned MyoHub myo_sim submodule for MS-51 (#10344).
# .gitmodules records the URL/path; the gitlink SHA is the pin of record.
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PinSha = "33f3ded946f55adbdcf963c99999587aadaf975f"
$SubmodulePath = "shared/models/myosuite/myo_sim"

Set-Location $Root

Write-Host "Initializing $SubmodulePath at pin $PinSha ..."
git submodule update --init -- $SubmodulePath
if ($LASTEXITCODE -ne 0) {
    throw "git submodule update failed for $SubmodulePath"
}

$Current = (git -C $SubmodulePath rev-parse HEAD).Trim()
if (-not $Current.StartsWith($PinSha.Substring(0, [Math]::Min(12, $PinSha.Length))) -and $Current -ne $PinSha) {
    Write-Host "Checking out recorded pin $PinSha (was $Current) ..."
    git -C $SubmodulePath fetch --depth 1 origin $PinSha 2>$null
    git -C $SubmodulePath checkout $PinSha
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to checkout myo_sim pin $PinSha"
    }
}

$Body = Join-Path $Root "$SubmodulePath/body/myobody_simpleupper.xml"
if (-not (Test-Path $Body)) {
    throw "Expected $Body after checkout"
}

Write-Host "Generating golfer scenes ..."
python -c @"
from pathlib import Path
from src.engines.physics_engines.myosuite.python.golfer_scene import (
    MYO_SIM_PIN_SHA,
    generate_golfer_scene,
)

paths = generate_golfer_scene(repo_root=Path(r'$Root'))
print(f'pin={MYO_SIM_PIN_SHA}')
print(f'driver={paths.driver}')
print(f'iron={paths.iron}')
print(f'receipt={paths.receipt}')
"@

Write-Host "MyoSuite models ready."
