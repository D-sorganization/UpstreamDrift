import re
from pathlib import Path

content = Path("src/shared/python/core/contracts/decorators.py").read_text("utf-8")

# Remove conflict markers
content = re.sub("=" * 7 + "\n", "", content)
content = re.sub("<" * 7 + " HEAD\n", "", content)
content = re.sub(">" * 7 + " origin/main\n", "", content)

Path("src/shared/python/core/contracts/decorators.py").write_text(content, "utf-8")
