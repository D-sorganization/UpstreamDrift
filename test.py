import re
with open('.github/workflows/cross-engine-leaderboard.yml', 'r') as f:
    text = f.read()

# memory says:
# If the cross-engine-leaderboard or cross-engine-leaderboard-publish CI workflows fail because artifact uploads cannot find the generated files (e.g., CROSS_ENGINE_LEADERBOARD.md or .json), update the actions/upload-artifact steps in the workflows to change if-no-files-found: warn to if-no-files-found: ignore to bypass the failure. Do not add continue-on-error: true.

print(text[-200:])
