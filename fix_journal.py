import re
from datetime import date

today = date.today().strftime("%Y-%m-%d")

with open(".jules/bolt.md", "r") as f:
    content = f.read()

# Replace literal `$(date +%Y-%m-%d)` with actual date
content = content.replace("$(date +%Y-%m-%d)", today)

# We want to deduplicate the journal entries if they were accidentally appended twice
entries = content.split(f"## {today} - Optimization of np.linalg.norm with keepdims using einsum")
if len(entries) > 2:
    # There's more than one copy!
    fixed_content = f"## {today} - Optimization of np.linalg.norm with keepdims using einsum".join(entries[:2])
    with open(".jules/bolt.md", "w") as f:
        f.write(fixed_content)
else:
    with open(".jules/bolt.md", "w") as f:
        f.write(content)
