from pathlib import Path

from scripts.ci.check_tutorial_imports import (
    _repo_root,
    check_doc_imports,
    extract_python_blocks,
    iter_import_from_modules,
)


def test_extract_python_blocks_returns_fenced_python_only() -> None:
    markdown = """\\```python
from pathlib import Path
\\```

```bash
python -m pytest
```
"""

    assert extract_python_blocks(markdown) == ["from pathlib import Path\n"]


def test_iter_import_from_modules_collects_from_imports() -> None:
    source = (
        "import os\n"
        "from pathlib import Path\n"
        "from src.shared.python.engine_core.engine_manager import EngineManager\n"
    )

    assert iter_import_from_modules(source) == [
        "pathlib",
        "src.shared.python.engine_core.engine_manager",
    ]


def test_check_doc_imports_reports_deprecated_engine_manager_import(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "broken.md"
    doc.write_text(
        (
            "```python\n"
            "from src.shared.python.engine_manager import EngineManager\n"
            "```\n"
        ),
        encoding="utf-8",
    )

    errors = check_doc_imports((doc,))

    assert errors
    assert "src.shared.python.engine_manager" in "\n".join(errors)


def test_tutorial_imports_script_resolves_repo_root() -> None:
    assert _repo_root() == Path(__file__).resolve().parents[2]
