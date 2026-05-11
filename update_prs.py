import subprocess
import json

res = subprocess.run(["gh", "pr", "list", "--state", "open", "--json", "number"], capture_output=True, text=True)
if res.returncode == 0 and res.stdout:
    prs = json.loads(res.stdout)
    for pr in prs:
        print(f"Updating PR {pr['number']}")
        subprocess.run(["gh", "pr", "update-branch", str(pr["number"])])
