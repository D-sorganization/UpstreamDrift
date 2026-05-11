import subprocess
import json

res = subprocess.run(
    ["gh", "pr", "list", "--state", "open", "--json", "number"],
    capture_output=True,
    text=True,
)
if res.returncode == 0 and res.stdout:
    prs = json.loads(res.stdout)
    for pr in prs:
        subprocess.run(["gh", "pr", "merge", str(pr["number"]), "--squash", "--auto"])
        print(  # noqa: T201
            f"Set auto-merge for PR {pr['number']}"
        )
