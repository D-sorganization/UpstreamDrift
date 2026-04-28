$files = Get-ChildItem -Path .github/workflows/ -Filter "Jules-*.yml"

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    
    # Skip if already has kill switch
    if ($content -match "check-kill-switch") {
        Write-Host "Skipping $($file.Name) - already has kill switch"
        continue
    }

    # Skip if it doesn't have pick-runner
    if ($content -notmatch "pick-runner:") {
        Write-Host "Skipping $($file.Name) - no pick-runner job"
        continue
    }

    Write-Host "Processing $($file.Name)..."

    # 1. Remove conflict markers if any
    $content = $content -replace "<<<<<<< HEAD`r?`n", ""
    $content = $content -replace "=======`r?`n", ""
    $content = $content -replace ">>>>>>> origin/main`r?`n", ""
    
    # 2. Add check-kill-switch job
    $killSwitchJob = @"
  check-kill-switch:
    runs-on: ubuntu-latest
    outputs:
      enabled: `${{ steps.check.outputs.enabled }}
    steps:
      - name: Check Jules Kill Switch
        id: check
        run: |
          if [ "`${{ vars.JULES_ENABLED }}" = "false" ]; then
            echo "Jules is DISABLED via kill switch."
            echo "enabled=false" >> `$GITHUB_OUTPUT
          else
            echo "enabled=true" >> `$GITHUB_OUTPUT
          fi

"@
    $content = $content -replace "(?m)^jobs:\s*\n", "jobs:`n$killSwitchJob"

    # 3. Update pick-runner job
    $content = $content -replace "(?m)^  pick-runner:\s*\n    runs-on: d-sorg-fleet", "  pick-runner:`n    needs: check-kill-switch`n    if: needs.check-kill-switch.outputs.enabled == 'true'`n    runs-on: d-sorg-fleet"

    # 4. Update worker jobs (any job that needs pick-runner)
    $content = [regex]::Replace($content, "(?m)^  (\w+):\s*\n    needs: pick-runner", "  `$1:`n    needs: [check-kill-switch, pick-runner]`n    if: needs.check-kill-switch.outputs.enabled == 'true'")
    
    # 5. Fix runs-on for worker jobs (should be self-hosted)
    $content = [regex]::Replace($content, "(?m)^  (\w+):\s*\n    needs: \[check-kill-switch, pick-runner\]\s*\n    if: needs.check-kill-switch.outputs.enabled == 'true'\s*\n    runs-on: ubuntu-latest", "  `$1:`n    needs: [check-kill-switch, pick-runner]`n    if: needs.check-kill-switch.outputs.enabled == 'true'`n    runs-on: self-hosted")

    Set-Content -Path $file.FullName -Value $content
}
