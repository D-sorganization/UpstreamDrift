import re
from pathlib import Path

content = Path("src/shared/python/core/contracts/decorators.py").read_text("utf-8")

# Remove conflict markers
content = re.sub(r"=======\n", "", content)
content = re.sub(r"<<<<<<< HEAD\n", "", content)
content = re.sub(r">>>>>>> origin/main\n", "", content)

Path("src/shared/python/core/contracts/decorators.py").write_text(content, "utf-8")
