import os
import re

workflow_dir = ".github/workflows"
fixed_files = []

# Hardened pick-runner logic
hardened_replacement = """  pick-runner:
    runs-on: d-sorg-fleet
    timeout-minutes: 2
    outputs:
      runner: ${{ steps.check.outputs.runner }}
    steps:
      - name: Check for self-hosted runner
        id: check
        env:
          GH_TOKEN: ${{ secrets.RUNNER_CHECK_TOKEN }}
        run: |
          ONLINE=$(gh api -H "Accept: application/vnd.github+json" /orgs/${{ github.repository_owner }}/actions/runners --paginate \\
            --jq \'if .runners then [.runners[] | select(.status == "online") | select(.labels[].name == "d-sorg-fleet")] | length else 0 end\' \\
            2>/dev/null || echo "0")
          if [[ "$ONLINE" -gt 0 ]]; then
            echo "runner=d-sorg-fleet" >> $GITHUB_OUTPUT
            echo "Self-hosted runner online - routing locally"
          else
            echo "::error::No local self-hosted runner available; failing closed"
            exit 1
          fi"""

for filename in os.listdir(workflow_dir):
    if filename.endswith((".yml", ".yaml")):
        path = os.path.join(workflow_dir, filename)
        with open(path, encoding="utf-8") as f:
            content = f.read()

        # 1. Replace the whole pick-runner job
        new_content = re.sub(
            r"  pick-runner:[\s\S]*?(?=\n  [a-zA-Z]|\Z)", hardened_replacement, content
        )

        # 2. Fix encoding/dashes
        new_content = new_content.replace("—", " - ").replace("–", " - ")

        if new_content != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            fixed_files.append(filename)
