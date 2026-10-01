param(
  [Parameter(Mandatory)][string]$Script,
  [Parameter(Mandatory)][string]$Log,
  [int]$TimeoutMin = 40,
  [int]$DoneGraceSec = 90,
  [string]$Matlab = 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe'
)
# Runs one MATLAB -batch script at a time: GS3DX simulations share the
# Simulink cache, so parallel processes corrupt each other's runs.
#
#   .\run_matlab_locked.ps1 -Script C:\scratch\probe.m -Log C:\scratch\probe.log -TimeoutMin 120
#
# The script runs from its own folder.  It must end with
#   fprintf("GS3DX_BATCH_DONE\n"); quit(0, "force");
# because -batch MATLAB often hangs at exit: once that marker is in the log
# the process tree is killed after DoneGraceSec.  The log ends with "EXIT n"
# (124 = killed by the TimeoutMin watchdog).  Set CAPTURE_DATA_DIR in the
# environment first when the script reads the capture data.
#
# First come, first served: each waiter drops a ticket in the queue folder;
# the oldest ticket whose owner is alive takes the lock next.  A lock whose
# owner died is cleared.  Never kill MATLAB processes this runner did not
# start.
$lock = Join-Path $env:USERPROFILE 'gs3dx_matlab.lock'
$queue = Join-Path $env:USERPROFILE 'gs3dx_matlab.queue'
New-Item -ItemType Directory -Force -Path $queue | Out-Null
$ticket = Join-Path $queue ('{0:D19}_{1}' -f [DateTime]::UtcNow.Ticks, $PID)
Set-Content -Path $ticket -Value $Script
try {
  while ($true) {
    $first = Get-ChildItem $queue | Sort-Object Name | Where-Object {
      $owner = [int]($_.Name -split '_')[1]
      if (Get-Process -Id $owner -ErrorAction SilentlyContinue) { $true } else { Remove-Item $_.FullName -ErrorAction SilentlyContinue; $false }
    } | Select-Object -First 1
    $ownerFile = "$lock\owner.txt"
    if (Test-Path $ownerFile) {
      $lockOwner = [int]((Get-Content $ownerFile -ErrorAction SilentlyContinue | Select-Object -First 1) -split ' ')[0]
      if ($lockOwner -and -not (Get-Process -Id $lockOwner -ErrorAction SilentlyContinue)) {
        Remove-Item -Recurse -Force $lock -ErrorAction SilentlyContinue
      }
    }
    if ($first -and $first.FullName -eq $ticket) {
      try { New-Item -ItemType Directory -Path $lock -ErrorAction Stop | Out-Null; break } catch { }
    }
    Start-Sleep -Seconds 10
  }
} finally { Remove-Item $ticket -ErrorAction SilentlyContinue }
try {
  Set-Content -Path "$lock\owner.txt" -Value "$PID $Script $(Get-Date -Format o)"
  $dir = Split-Path $Script; $name = Split-Path $Script -Leaf
  $err = "$Log.stderr"
  $p = Start-Process -FilePath $Matlab -ArgumentList '-batch', "run('$name')" `
       -WorkingDirectory $dir -RedirectStandardOutput $Log -RedirectStandardError $err -NoNewWindow -PassThru
  $null = $p.Handle   # keep the handle so ExitCode is available after exit
  $deadline = (Get-Date).AddMinutes($TimeoutMin)
  $doneAt = $null
  $killed = $null
  while (-not $p.WaitForExit(15000)) {
    if ((Get-Date) -gt $deadline) { $killed = 'timeout'; break }
    if (-not $doneAt -and (Test-Path $Log) -and (Select-String -Path $Log -SimpleMatch 'GS3DX_BATCH_DONE' -Quiet)) { $doneAt = Get-Date }
    if ($doneAt -and ((Get-Date) - $doneAt).TotalSeconds -gt $DoneGraceSec) { $killed = 'done'; break }
  }
  if ($killed) {
    taskkill /PID $p.Id /T /F | Out-Null
    if ($killed -eq 'done') {
      "RUNNER: script finished (GS3DX_BATCH_DONE); MATLAB hung at exit and was killed after $DoneGraceSec s" | Add-Content $Log
      "EXIT 0" | Add-Content $Log
    } else {
      "WATCHDOG: killed after $TimeoutMin min (script hung; do not open Mechanics Explorer or GUI windows under -batch)" | Add-Content $Log
      "EXIT 124" | Add-Content $Log
    }
  } else {
    if (Test-Path $err) { Get-Content $err | Add-Content $Log }
    "EXIT $($p.ExitCode)" | Add-Content $Log
  }
} finally { Remove-Item -Recurse -Force $lock -ErrorAction SilentlyContinue }
