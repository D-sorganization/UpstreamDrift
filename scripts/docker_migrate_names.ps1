<#
.SYNOPSIS
Migration script for legacy UpstreamDrift Docker image names to canonical tags.

.DESCRIPTION
Provides a safe, reversible retagging utility for transitioning legacy 
local Docker image names (robotics_env, golf-suite, etc.) to the new
unified 'upstream-drift:*' schema without deleting images.

.PARAMETER DryRun
If specified, the script only prints what it would do.

.PARAMETER Rollback
If specified, performs the reverse transition from canonical back to legacy tags.
#>

param(
    [switch]$DryRun = $false,
    [switch]$Rollback = $false
)

$ErrorActionPreference = "Stop"

function Confirm-ImageExists($imageName) {
    try {
        $out = docker image inspect $imageName 2>&1
        if ($LASTEXITCODE -eq 0 -or $out -match "Id") {
            return $true
        }
    } catch {}
    return $false
}

function Tag-Image($source, $target) {
    if (Confirm-ImageExists $source) {
        if ($DryRun) {
            Write-Host "[DRY RUN] Would tag '$source' -> '$target'" -ForegroundColor Cyan
        } else {
            Write-Host "Tagging '$source' -> '$target'..." -ForegroundColor Green
            docker tag $source $target
        }
    } else {
        Write-Host "Source image '$source' not found locally. Skipping." -ForegroundColor DarkGray
    }
}

Write-Host "=== UpstreamDrift Docker Name Migration ===" -ForegroundColor Yellow
if ($DryRun) { Write-Host "MODE: Dry Run" -ForegroundColor Yellow }
if ($Rollback) { Write-Host "MODE: Rollback (Reverse)" -ForegroundColor Yellow }
Write-Host "-------------------------------------------"

if ($Rollback) {
    Tag-Image "upstream-drift:engine" "robotics_env:latest"
    Tag-Image "upstream-drift:runtime" "golf-suite:latest"
    Tag-Image "upstream-drift:dev" "golf-suite-dev:latest"
} else {
    Tag-Image "robotics_env:latest" "upstream-drift:engine"
    Tag-Image "golf-suite:latest" "upstream-drift:runtime"
    Tag-Image "golf-suite-dev:latest" "upstream-drift:dev"
}

Write-Host "Done."
