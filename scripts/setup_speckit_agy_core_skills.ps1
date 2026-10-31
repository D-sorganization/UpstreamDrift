# Materialize Spec Kit core Antigravity skills from the pinned Specify CLI.
# Git extension skills are repo-owned under .agents/skills/speckit-git-*.
$ErrorActionPreference = "Stop"
Set-Location (git rev-parse --show-toplevel)
if (-not (Get-Command specify -ErrorAction SilentlyContinue)) {
    Write-Error "specify CLI not found; install with: pip install specify-cli"
}
specify integration install agy --force --trust-integration
