<#
.SYNOPSIS
  Run or replay a Simscape matched-swing candidate with an R2025b receipt (MS-60).

.DESCRIPTION
  One documented entry point for Simscape candidate replay on a licensed host.
  Fail-closed: refuses non-R2025b MATLAB. Default CI without a license must not
  treat this as a green native qualification.

.PARAMETER Run
  Evidence run id under native_evidence/ (default: two_window_fit_9967_102).

.PARAMETER Replay
  Invoke MATLAB R2025b -batch replay_returned102_r2025b for the run.

.PARAMETER MatlabExe
  Explicit matlab.exe path. Defaults to R2025b under Program Files.

.PARAMETER RepoRoot
  Repository root. Defaults to the checkout containing this script.

.PARAMETER RefreshPythonCandidate
  After validation, refresh candidate.npz via Python converter + candidate_io.

.EXAMPLE
  powershell scripts/matlab/run_simscape_candidate.ps1 -Run two_window_fit_9967_102 -Replay
#>
[CmdletBinding()]
param(
    [string]$Run = "two_window_fit_9967_102",
    [switch]$Replay,
    [string]$MatlabExe = "",
    [string]$RepoRoot = "",
    [switch]$RefreshPythonCandidate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    param([string]$Hint)
    if ($Hint) { return (Resolve-Path $Hint).Path }
    $here = Split-Path -Parent $PSCommandPath
    return (Resolve-Path (Join-Path $here "..\..")).Path
}

function Resolve-MatlabR2025b {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path $Explicit)) {
            throw "MatlabExe not found: $Explicit"
        }
        return (Resolve-Path $Explicit).Path
    }
    $candidates = @(
        "${env:ProgramFiles}\MATLAB\R2025b\bin\matlab.exe",
        "${env:ProgramFiles}\MATLAB\R2025b\bin\win64\matlab.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    throw "MATLAB R2025b matlab.exe not found. Pass -MatlabExe explicitly. No R2026a substitution."
}

function Get-FileSha256Hex {
    param([string]$Path)
    $hash = Get-FileHash -Algorithm SHA256 -Path $Path
    return $hash.Hash.ToLowerInvariant()
}

$repo = Resolve-RepoRoot -Hint $RepoRoot
$evidenceRel = "docs/development/simscape_tour_matching/native_evidence/$Run"
$evidenceDir = Join-Path $repo $evidenceRel
if (-not (Test-Path $evidenceDir)) {
    throw "Evidence directory missing: $evidenceDir"
}

$replayNpz = Join-Path $evidenceDir "returned-replay.npz"
$candidateJson = Join-Path $evidenceDir "returned-candidate.json"
$qualifiedJson = Join-Path $evidenceDir "qualified_candidate_replay.json"
foreach ($required in @($replayNpz, $candidateJson)) {
    if (-not (Test-Path $required)) {
        throw "Required evidence file missing: $required"
    }
}

$wallClock = 0.0
$hostName = $env:COMPUTERNAME
if (-not $hostName) { $hostName = [System.Net.Dns]::GetHostName() }

if ($Replay) {
    if ($Run -ne "two_window_fit_9967_102") {
        throw "Replay only implemented for two_window_fit_9967_102; received -Run $Run"
    }
    $matlab = Resolve-MatlabR2025b -Explicit $MatlabExe
    $sharedMatlab = Join-Path $repo "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared"
    $batch = @"
addpath(genpath('$($sharedMatlab -replace '\\','/')'));
addpath('$($evidenceDir -replace '\\','/')');
repo = '$($repo -replace '\\','/')';
tic;
report = replay_returned102_r2025b(repo);
elapsed = toc;
assert(strcmpi(string(report.matlab_release), "2025b") || strcmpi(string(report.matlab_release), "R2025b"), ...
    'R2025b required');
cand = jsondecode(fileread(fullfile(repo, '$($evidenceRel -replace '\\','/')', 'returned-candidate.json')));
receipt = jsondecode(fileread(fullfile(repo, '$($evidenceRel -replace '\\','/')', 'receipt.json')));
fields = struct();
fields.run_id = '$Run';
fields.matlab_release = char(report.matlab_release);
fields.matlab_version = char(report.matlab_version);
fields.host = '$hostName';
fields.machine = '$hostName';
fields.model_sha256 = char(cand.model_sha256);
fields.candidate_sha256 = char(receipt.returned_sha256);
fields.replay_npz_sha256 = lower('$(Get-FileSha256Hex $replayNpz)');
fields.wall_clock_s = elapsed;
fields.qualification = char(report.qualification);
fields.evidence_dir = '$($evidenceRel -replace '\\','/')';
fields.artifacts = struct( ...
    'candidate_npz', 'candidate.npz', ...
    'playback_gif', 'playback.gif', ...
    'returned_replay_npz', 'returned-replay.npz', ...
    'qualified_replay_json', 'qualified_candidate_replay.json');
fields.issue = '#10347';
write_run_manifest(fullfile(repo, '$($evidenceRel -replace '\\','/')', 'run_manifest.json'), fields);
export_candidate(fullfile(repo, '$($evidenceRel -replace '\\','/')'));
disp(jsonencode(report.metrics));
"@
    Write-Host "Invoking R2025b batch replay for $Run ..."
    & $matlab -batch $batch
    if ($LASTEXITCODE -ne 0) {
        throw "MATLAB R2025b replay failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "Replay skipped. Validating committed receipts for $Run ..."
    if (-not (Test-Path (Join-Path $evidenceDir "run_manifest.json"))) {
        throw "run_manifest.json missing; re-run with -Replay on a licensed R2025b host or restore committed receipt."
    }
    if (-not (Test-Path (Join-Path $evidenceDir "candidate.npz"))) {
        throw "candidate.npz missing; refresh with -RefreshPythonCandidate or restore committed package."
    }
}

if ($RefreshPythonCandidate) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) { throw "python not on PATH for -RefreshPythonCandidate" }
    $code = @"
import json
from pathlib import Path
from src.shared.python.motion_matching.candidate_convert import convert_simscape_returned_replay
from src.shared.python.motion_matching.candidate_io import save_candidate
evidence = Path(r'$evidenceDir')
doc = json.loads((evidence / 'returned-candidate.json').read_text(encoding='utf-8'))
cand = convert_simscape_returned_replay(evidence / 'returned-replay.npz', candidate_doc=doc, engine='simscape')
save_candidate(cand, evidence / 'candidate.npz')
print('refreshed', evidence / 'candidate.npz')
"@
    Push-Location $repo
    try {
        & python -c $code
        if ($LASTEXITCODE -ne 0) { throw "Python candidate refresh failed" }
    } finally {
        Pop-Location
    }
}

Write-Host "MS-60 run management OK for $Run (host=$hostName)."
if (Test-Path $qualifiedJson) {
    Write-Host "Qualified receipt present: $qualifiedJson"
}
