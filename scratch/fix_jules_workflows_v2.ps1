$files = Get-ChildItem -Path .github/workflows/ -Filter "Jules-*.yml"

foreach ($file in $files) {
    # Skip jules-kill-switch itself
    if ($file.Name -eq "jules-kill-switch.yml") { continue }
    # Skip Jules-Control-Tower as it's already fixed
    if ($file.Name -eq "Jules-Control-Tower.yml") { continue }
    
    Write-Host "Processing $($file.Name)..."
    $content = Get-Content $file.FullName -Raw
    
    # 1. Clean conflict markers if present
    $content = $content -replace "<<<<<<< HEAD`r?`n", ""
    $content = $content -replace "=======`r?`n", ""
    $content = $content -replace ">>>>>>> origin/main`r?`n", ""
    
    # 2. Extract preamble (name, on, permissions, env, concurrency)
    if ($content -match "(?s)^(.*?)\njobs:") {
        $preamble = $matches[1]
    } else {
        Write-Host "  Could not find jobs section in $($file.Name)"
        continue
    }

    # 3. Extract the 'real' work steps
    # We look for steps after the conflict markers or in the main job
    # For many of these, the 'pick-runner' was merged into the main job or vice versa
    
    # Find the steps that aren't the pick-runner check
    if ($content -match "(?s)steps:.*?- uses: actions/checkout@v6(.*?)$") {
        $workSteps = "      - uses: actions/checkout@v6" + $matches[1]
    } elseif ($content -match "(?s)steps:.*?- name: Checkout(.*?)$") {
        $workSteps = "      - name: Checkout" + $matches[1]
    } else {
        Write-Host "  Could not find work steps in $($file.Name)"
        continue
    }

    # 4. Construct the clean workflow
    $jobName = ($file.BaseName -replace "Jules-", "").ToLower() -replace "[^a-z0-9]", "-"
    
    $newContent = $preamble + "`n" + @"
jobs:
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

  pick-runner:
    needs: check-kill-switch
    if: needs.check-kill-switch.outputs.enabled == 'true'
    runs-on: d-sorg-fleet
    timeout-minutes: 2
    outputs:
      runner: `${{ steps.check.outputs.runner }}
    steps:
      - name: Check for self-hosted runner
        id: check
        env:
          GH_TOKEN: `${{ secrets.RUNNER_CHECK_TOKEN }}
        run: |
          ONLINE=$(gh api -H "Accept: application/vnd.github+json" /orgs/`${{ github.repository_owner }}/actions/runners --paginate \
            --jq 'if .runners then [.runners[] | select(.status == "online") | select(.labels[].name == "d-sorg-fleet")] | length else 0 end' \
            2>/dev/null || echo "0")
          if [[ "`$ONLINE" =~ ^[0-9]+$ ]] && [[ "`$ONLINE" -gt 0 ]]; then
            echo "runner=d-sorg-fleet" >> `$GITHUB_OUTPUT
            echo "Self-hosted runner online - routing locally"
          else
            echo "::error::No local self-hosted runner available; failing closed"
            exit 1
          fi

  $jobName:
    needs: [check-kill-switch, pick-runner]
    if: needs.check-kill-switch.outputs.enabled == 'true'
    runs-on: self-hosted
    steps:
$workSteps
"@
    
    Set-Content -Path $file.FullName -Value $newContent
}
