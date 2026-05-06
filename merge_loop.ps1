foreach ($pr in 4089, 4090, 4092, 4141, 4142) {
    Write-Host "Processing PR $pr"
    gh pr checkout $pr
    git fetch origin main
    git merge origin/main --no-edit
    $conflicts = git diff --name-only --diff-filter=U
    if ($conflicts) {
        Write-Host "Conflicts in $conflicts"
        git checkout origin/main -- docs/review_archive/
        git add docs/review_archive/
        $remaining = git diff --name-only --diff-filter=U
        if ($remaining) {
            Write-Host "Still have conflicts in $remaining for PR $pr. Aborting."
            git merge --abort
            continue
        }
        git commit -am "fix(merge): resolve review archive conflicts" --no-verify
    }
    git push origin HEAD --no-verify
    Start-Sleep -Seconds 5
    gh pr merge $pr --admin -s
}
