from pathlib import Path

from scripts.check_test_layout import LEGACY_ROOT_TEST_FILES, audit_test_layout


def test_no_legacy_root_test_allowlist_remains() -> None:
    assert frozenset() == LEGACY_ROOT_TEST_FILES


def test_audit_test_layout_accepts_clean_tree(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests" / "launchers"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_launcher.py").write_text("def test_ok():\n    pass\n")
    (tmp_path / "src" / "launchers").mkdir(parents=True)

    assert audit_test_layout(tmp_path) == []


def test_audit_test_layout_rejects_root_test_files(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_flat.py").write_text("def test_flat():\n    pass\n")

    findings = audit_test_layout(tmp_path)

    assert len(findings) == 1
    assert findings[0].path == Path("tests/test_flat.py")
    assert "root-level test file" in findings[0].reason


def test_audit_test_layout_rejects_src_tests_directories(tmp_path: Path) -> None:
    src_tests = tmp_path / "src" / "launchers" / "tests"
    src_tests.mkdir(parents=True)

    findings = audit_test_layout(tmp_path)

    assert len(findings) == 1
    assert findings[0].path == Path("src/launchers/tests")
    assert "tests directory under src" in findings[0].reason


def test_audit_test_layout_rejects_overlapping_fixture_names(
    tmp_path: Path,
) -> None:
    root_conftest = tmp_path / "tests" / "conftest.py"
    unit_conftest = tmp_path / "tests" / "unit" / "conftest.py"
    unit_conftest.parent.mkdir(parents=True)
    root_conftest.write_text(
        "import pytest\n\n@pytest.fixture\ndef shared_fixture():\n    return 1\n",
        encoding="utf-8",
    )
    unit_conftest.write_text(
        "import pytest\n\n@pytest.fixture\ndef shared_fixture():\n    return 2\n",
        encoding="utf-8",
    )

    findings = audit_test_layout(tmp_path)

    assert len(findings) == 1
    assert findings[0].path == Path("tests/unit/conftest.py")
    assert "duplicate fixture" in findings[0].reason
