#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# GAAI Daemon Launcher for Golf_GAAI_Sandbox
#
# Run this script from WSL to start the autonomous overnight delivery daemon.
#
# Prerequisites (run once in WSL):
#   sudo apt install -y tmux nodejs npm
#   npm install -g @anthropic-ai/claude-code
#   claude --version   # verify claude is in PATH
#
# Usage:
#   From Windows Git Bash or WSL:
#     bash start-gaai-daemon.sh            # interactive (status output)
#     bash start-gaai-daemon.sh --dry-run  # show what would launch, no action
#     bash start-gaai-daemon.sh --status   # check active sessions
#
# Monitor overnight:
#   tmux attach -t gaai-daemon
#   (Ctrl+B then D to detach without stopping)
#
# View individual story logs:
#   tail -f .gaai/project/contexts/backlog/.delivery-logs/E01S01.log
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_SCRIPT="$REPO_ROOT/.gaai/core/scripts/delivery-daemon.sh"

# ── Preflight checks ──────────────────────────────────────────────────────────
check_dependency() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: '$1' not found. Install with: $2"
    exit 1
  fi
}

check_dependency "tmux"   "sudo apt install tmux -y"
check_dependency "claude" "npm install -g @anthropic-ai/claude-code"
check_dependency "git"    "sudo apt install git -y"

# ── Suppress dangerous-mode permission prompt ─────────────────────────────────
CLAUDE_SETTINGS="$HOME/.claude/settings.json"
if [[ ! -f "$CLAUDE_SETTINGS" ]] || ! grep -q "skipDangerousModePermissionPrompt" "$CLAUDE_SETTINGS" 2>/dev/null; then
  echo "Configuring claude to skip dangerous-mode permission prompt..."
  mkdir -p "$HOME/.claude"
  echo '{ "skipDangerousModePermissionPrompt": true }' > "$CLAUDE_SETTINGS"
  echo "  -> Written to $CLAUDE_SETTINGS"
fi

# ── Ensure we're on staging branch ───────────────────────────────────────────
cd "$REPO_ROOT"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "staging" ]]; then
  echo "WARNING: Not on staging branch (currently on '$CURRENT_BRANCH')."
  echo "Switching to staging..."
  git checkout staging
  git pull --ff-only origin staging
fi

# ── Pull latest before starting ──────────────────────────────────────────────
echo "Fetching latest from origin/staging..."
git fetch origin
git pull --ff-only origin staging || echo "Note: fast-forward failed (may have local changes)"

# ── Pass through arguments to daemon ─────────────────────────────────────────
ARGS="${*}"

if [[ "$ARGS" == *"--status"* ]]; then
  bash "$DAEMON_SCRIPT" --status
  exit 0
fi

if [[ "$ARGS" == *"--dry-run"* ]]; then
  echo "DRY RUN — stories that would be launched:"
  bash "$DAEMON_SCRIPT" --dry-run
  exit 0
fi

# ── Launch daemon in tmux session ─────────────────────────────────────────────
SESSION_NAME="gaai-daemon"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "GAAI daemon session '$SESSION_NAME' is already running."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  tmux attach -t "$SESSION_NAME"
  exit 0
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║       GAAI Autonomous Delivery Daemon            ║"
echo "║       Golf_GAAI_Sandbox — Overnight Run          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Starting daemon in tmux session '$SESSION_NAME'..."
echo "Concurrent stories: 2 (conservative for overnight run)"
echo "Poll interval: 60 seconds"
echo ""

# Launch daemon: 2 concurrent max for overnight safety, 60s poll
tmux new-session -d -s "$SESSION_NAME" \
  "bash $DAEMON_SCRIPT --max-concurrent 2 --interval 60; read -p 'Daemon exited. Press Enter to close.'"

echo "Daemon started in tmux session '$SESSION_NAME'."
echo ""
echo "Useful commands:"
echo "  tmux attach -t $SESSION_NAME           # watch live"
echo "  Ctrl+B then D                           # detach without stopping"
echo "  bash start-gaai-daemon.sh --status      # check what's running"
echo "  tail -f .gaai/project/contexts/backlog/.delivery-logs/*.log  # story logs"
echo ""
echo "Attaching now... (Ctrl+B D to detach)"
sleep 2
tmux attach -t "$SESSION_NAME"
