#!/usr/bin/env bash
# Run AFTER `gh auth refresh -h github.com -s project,read:project`.
#
# Creates the "URDF Hardening Campaign" GitHub Project under
# D-sorganization and bulk-adds all milestone-1 issues to it.
set -euo pipefail

OWNER="D-sorganization"
REPO="D-sorganization/UpstreamDrift"
TITLE="URDF Hardening Campaign"

echo "Creating project '$TITLE' under $OWNER..."
PROJECT_JSON=$(gh project create --owner "$OWNER" --title "$TITLE" --format json)
PROJECT_NUMBER=$(echo "$PROJECT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['number'])")
echo "Created project #$PROJECT_NUMBER"

echo "Fetching milestone-1 issues..."
ISSUES=$(gh issue list --repo "$REPO" --state open --search 'milestone:"URDF Hardening Campaign"' --limit 100 --json number,url)
COUNT=$(echo "$ISSUES" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
echo "Found $COUNT issues to add."

echo "$ISSUES" | python3 -c "import json,sys; [print(i['url']) for i in json.load(sys.stdin)]" | while read -r URL; do
  echo "  + $URL"
  gh project item-add "$PROJECT_NUMBER" --owner "$OWNER" --url "$URL" >/dev/null
done

echo
echo "Done. Project URL:"
echo "$PROJECT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['url'])"
