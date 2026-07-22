import re

file_path = ".github/workflows/cross-engine-leaderboard.yml"
with open(file_path, "r") as f:
    content = f.read()

content = content.replace("        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a\n        with:\n          name: cross-engine-leaderboard\n          path: motion_matching/results/CROSS_ENGINE_LEADERBOARD.md\n          if-no-files-found: ignore\n        continue-on-error: true", "        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a\n        continue-on-error: true\n        with:\n          name: cross-engine-leaderboard\n          path: motion_matching/results/CROSS_ENGINE_LEADERBOARD.md\n          if-no-files-found: ignore")

with open(file_path, "w") as f:
    f.write(content)

file_path2 = ".github/workflows/cross-engine-leaderboard-publish.yml"
with open(file_path2, "r") as f:
    content2 = f.read()

content2 = content2.replace("        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a\n        with:\n          name: cross-engine-leaderboard-json\n          path: reports/cross_engine_leaderboard.json\n          if-no-files-found: ignore\n        continue-on-error: true", "        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a\n        continue-on-error: true\n        with:\n          name: cross-engine-leaderboard-json\n          path: reports/cross_engine_leaderboard.json\n          if-no-files-found: ignore")

with open(file_path2, "w") as f:
    f.write(content2)
