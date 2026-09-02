import sys
import re
from datetime import datetime

filepath = "SPEC.md"
with open(filepath, "r") as f:
    content = f.read()

# Find current Spec Version
version_match = re.search(r"\|\s*\*\*Spec Version\*\*\s*\|\s*([0-9]+\.[0-9]+\.[0-9]+)\s*\|", content)
if not version_match:
    print("Warning: Could not find Spec Version.")
    sys.exit(1)

current_version = version_match.group(1)
parts = current_version.split('.')
new_version = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"

# Update Spec Version
content = content.replace(
    f"| **Spec Version**        | {current_version}",
    f"| **Spec Version**        | {new_version}"
)

# Insert Changelog entry at the top of the table
date_str = datetime.now().strftime("%Y-%m-%d")
changelog_entry = f"| {date_str} | {new_version} | Optimized collision generator vertex magnitude calculation (spec-exempt: micro-optimization) |\n"

# Find the start of the changelog table and insert
table_header = "| Date       | Version | Changes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |\n| ---------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |\n"
if table_header in content:
    content = content.replace(table_header, table_header + changelog_entry)
else:
    print("Warning: Could not find changelog table header.")
    sys.exit(1)

with open(filepath, "w") as f:
    f.write(content)
