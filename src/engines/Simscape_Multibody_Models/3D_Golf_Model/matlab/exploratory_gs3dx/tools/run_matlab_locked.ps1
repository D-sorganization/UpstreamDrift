param(
  [Parameter(Mandatory)][string]$Script,
  [Parameter(Mandatory)][string]$Log,
  [int]$TimeoutMin = 40,
  [int]$DoneGraceSec = 90,
  [string]$Matlab = 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe',
  [string]$LockDir = '',
  [string]$QueueDir = '',
  [double]$PollIntervalSec = 10
)

# Runs one MATLAB -batch script at a time: GS3DX simulations share the
# Simulink cache, so parallel processes corrupt each other's runs.
#
# Hardened contract:
# 1. GS3DX_BATCH_DONE is sanitized to exact complete lines (no comment/echo false positives).
# 2. Watchdog-killed hung exits never synthesize native EXIT 0; they return distinct code 125
#    (shutdown unverified) with actual_process_exit_code null in structured receipt.
# 3. Explicit non-success or unknown STATUS fails closed (returncode 1) even if process exited 0.
# 4. Timeout watchdog returns exit code 124.
# 5. Atomic lock ownership checks prevent deleting another live process's lock.
# 6. Locked paths are validated strictly within allowed directory boundary before recursive deletion.
# 7. Python decision helper is the single source of truth for evaluation; missing helper fails closed with 125.
# 8. Runner process itself exits with the final exit code matching the caller contract.

function Test-PathUnderAllowedRoot([string]$targetPath, [string]$allowedRoot) {
  if (-not $targetPath -or -not $allowedRoot) { return $false }
  $targetFull = [System.IO.Path]::GetFullPath($targetPath).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
  $rootFull = [System.IO.Path]::GetFullPath($allowedRoot).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)

  # Target cannot be the root itself
  if ($targetFull.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $false
  }

  # Sibling-prefix prevention: root + separator MUST be the prefix of targetFull
  $rootWithSep = $rootFull + [System.IO.Path]::DirectorySeparatorChar
  if (-not $targetFull.StartsWith($rootWithSep, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $false
  }
  return $true
}

function Remove-LockSafely([string]$targetLock, [string]$allowedRoot, [int]$expectedOwnerPid = 0) {
  if (-not (Test-Path -LiteralPath $targetLock)) { return }
  if (-not (Test-PathUnderAllowedRoot $targetLock $allowedRoot)) {
    Write-Warning "Refusing to remove lock at unsafe path: $targetLock"
    return
  }
  $ownerFile = Join-Path $targetLock 'owner.txt'
  if (Test-Path -LiteralPath $ownerFile) {
    $rawOwner = (Get-Content -LiteralPath $ownerFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($rawOwner) {
      $currentOwnerPid = [int]($rawOwner -split ' ')[0]
      if ($expectedOwnerPid -gt 0 -and $currentOwnerPid -ne $expectedOwnerPid) {
        # Never delete another process's lock
        return
      }
      if ($expectedOwnerPid -eq 0) {
        # Stale lock removal: verify recorded owner is actually dead
        if ($currentOwnerPid -gt 0 -and (Get-Process -Id $currentOwnerPid -ErrorAction SilentlyContinue)) {
          return
        }
      }
    }
  }
  Remove-Item -LiteralPath $targetLock -Recurse -Force -ErrorAction SilentlyContinue
}

function Test-ExactDoneMarkerInFile([string]$filePath) {
  if (-not (Test-Path -LiteralPath $filePath)) { return $false }
  $lines = Get-Content -LiteralPath $filePath -ErrorAction SilentlyContinue
  if (-not $lines) { return $false }
  foreach ($line in $lines) {
    if ($line.Trim() -eq 'GS3DX_BATCH_DONE') {
      return $true
    }
  }
  return $false
}

# Resolve lock and queue directories
if ($LockDir) {
  $lock = [System.IO.Path]::GetFullPath($LockDir)
  $lockParent = Split-Path $lock
} else {
  $lockParent = $env:USERPROFILE
  $lock = Join-Path $lockParent 'gs3dx_matlab.lock'
}

if ($QueueDir) {
  $queue = [System.IO.Path]::GetFullPath($QueueDir)
} else {
  $queue = Join-Path $env:USERPROFILE 'gs3dx_matlab.queue'
}

New-Item -ItemType Directory -Force -Path $queue | Out-Null
$ticket = Join-Path $queue ('{0:D19}_{1}' -f [DateTime]::UtcNow.Ticks, $PID)
Set-Content -LiteralPath $ticket -Value $Script

try {
  while ($true) {
    $first = Get-ChildItem -LiteralPath $queue | Sort-Object Name | Where-Object {
      $owner = [int]($_.Name -split '_')[1]
      if (Get-Process -Id $owner -ErrorAction SilentlyContinue) {
        $true
      } else {
        Remove-Item -LiteralPath $_.FullName -ErrorAction SilentlyContinue
        $false
      }
    } | Select-Object -First 1

    $ownerFile = Join-Path $lock 'owner.txt'
    if (Test-Path -LiteralPath $ownerFile) {
      $lockOwner = [int]((Get-Content -LiteralPath $ownerFile -ErrorAction SilentlyContinue | Select-Object -First 1) -split ' ')[0]
      if ($lockOwner -and -not (Get-Process -Id $lockOwner -ErrorAction SilentlyContinue)) {
        Remove-LockSafely $lock $lockParent 0
      }
    }

    if ($first -and $first.FullName -eq $ticket) {
      try {
        New-Item -ItemType Directory -Path $lock -ErrorAction Stop | Out-Null
        break
      } catch { }
    }
    Start-Sleep -Seconds $PollIntervalSec
  }
} finally {
  Remove-Item -LiteralPath $ticket -ErrorAction SilentlyContinue
}

$finalExitCode = 125
try {
  Set-Content -LiteralPath "$lock\owner.txt" -Value "$PID $Script $(Get-Date -Format o)"
  $dir = Split-Path $Script
  if (-not $dir) { $dir = (Get-Location).Path }
  $name = Split-Path $Script -Leaf
  $err = "$Log.stderr"
  $logDir = Split-Path $Log
  if ($logDir -and -not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
  }

  $p = Start-Process -FilePath $Matlab -ArgumentList '-batch', "run('$name')" `
       -WorkingDirectory $dir -RedirectStandardOutput $Log -RedirectStandardError $err `
       -WindowStyle Hidden -PassThru
  if ($null -eq $p) {
    "RUNNER: failed to start process $Matlab" | Add-Content -LiteralPath $Log
    "EXIT 125" | Add-Content -LiteralPath $Log
    exit 125
  }
  $null = $p.Handle

  $deadline = (Get-Date).AddMinutes($TimeoutMin)
  $doneAt = $null
  $killed = $null

  while (-not $p.WaitForExit(1000)) {
    if ((Get-Date) -gt $deadline) {
      $killed = 'timeout'
      break
    }
    if (-not $doneAt -and (Test-ExactDoneMarkerInFile $Log)) {
      $doneAt = Get-Date
    }
    if ($doneAt -and ((Get-Date) - $doneAt).TotalSeconds -gt $DoneGraceSec) {
      $killed = 'hung_after_done'
      break
    }
  }

  $actualProcessExitCode = $null
  $terminationReason = 'natural_exit'

  if ($killed) {
    $terminationReason = $killed
    taskkill /PID $p.Id /T /F | Out-Null
    Start-Sleep -Milliseconds 200
  } else {
    $actualProcessExitCode = $p.ExitCode
    if ($actualProcessExitCode -eq $null) {
      $terminationReason = 'unverified_exit'
    }
  }

  # Delegate decision and structured receipt generation to pure Python helper (single source of decisions)
  $decisionScript = Join-Path $PSScriptRoot 'runner_decision.py'
  $pyCmd = if (Get-Command python3 -ErrorAction SilentlyContinue) { 'python3' } elseif (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { $null }

  $evaluated = $false
  if ($pyCmd -and (Test-Path -LiteralPath $decisionScript)) {
    $rawExitArg = if ($actualProcessExitCode -ne $null) { "$actualProcessExitCode" } else { "null" }
    $pyArgs = @(
      $decisionScript,
      '--log', $Log,
      '--reason', $terminationReason,
      '--command', "run('$name')",
      '--pid', "$($p.Id)",
      '--actual-exit', $rawExitArg,
      '--append-to-log'
    )
    if (Test-Path -LiteralPath $err) {
      $pyArgs += @('--stderr', $err)
    }

    & $pyCmd @pyArgs
    if ($null -ne $LASTEXITCODE) {
      $finalExitCode = $LASTEXITCODE
      $evaluated = $true
    }
  }

  if (-not $evaluated) {
    # Decision helper or runtime unavailable: fail closed with 125, preserve structured receipt
    $finalExitCode = 125
    $receipt = [ordered]@{
      actual_process_exit_code = if ($actualProcessExitCode -ne $null) { [int]$actualProcessExitCode } else { $null }
      final_exit_code = 125
      termination_reason = $terminationReason
      done_marker = (Test-ExactDoneMarkerInFile $Log)
      script_status = $null
      command_identity = "run('$name')"
      owned_pids = @([int]$p.Id)
      timestamp = (Get-Date -Format o)
    }
    $receiptJson = $receipt | ConvertTo-Json -Compress
    $receiptJson | Set-Content -LiteralPath "$Log.receipt.json" -ErrorAction SilentlyContinue
    "RUNNER: decision helper or runtime unavailable; failing closed with returncode 125" | Add-Content -LiteralPath $Log
    "RECEIPT: $receiptJson" | Add-Content -LiteralPath $Log
    "EXIT 125" | Add-Content -LiteralPath $Log
  }
} finally {
  Remove-LockSafely $lock $lockParent $PID
}

exit $finalExitCode
