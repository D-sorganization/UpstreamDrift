from __future__ import annotations

from pathlib import Path

from scripts.check_engine_loaders import audit_engine_loaders


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_audit_engine_loaders_accepts_fit_file_without_loader_imports(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path
        / "src"
        / "engines"
        / "physics_engines"
        / "drake"
        / "python"
        / "motion_matching"
        / "fit_swing.py",
        "from src.shared.python.motion_matching.types import ClubTarget\n",
    )

    assert audit_engine_loaders(tmp_path) == []


def test_audit_engine_loaders_accepts_canonical_loader_import(tmp_path: Path) -> None:
    _write(
        tmp_path
        / "src"
        / "engines"
        / "physics_engines"
        / "mujoco"
        / "python"
        / "motion_matching"
        / "fit_swing_mujoco.py",
        "from src.shared.python.motion_matching.load_club_target import load_club_target_excel\n",
    )

    assert audit_engine_loaders(tmp_path) == []


def test_audit_engine_loaders_rejects_forbidden_from_import(tmp_path: Path) -> None:
    _write(
        tmp_path
        / "src"
        / "engines"
        / "physics_engines"
        / "opensim"
        / "python"
        / "motion_matching"
        / "fit_swing_opensim.py",
        "from opensim_loaders import load_club_target_excel\n",
    )

    findings = audit_engine_loaders(tmp_path)

    assert len(findings) == 1
    assert findings[0].path == Path(
        "src/engines/physics_engines/opensim/python/motion_matching/fit_swing_opensim.py"
    )
    assert "opensim_loaders" in findings[0].reason


def test_audit_engine_loaders_rejects_forbidden_import(tmp_path: Path) -> None:
    _write(
        tmp_path
        / "src"
        / "engines"
        / "physics_engines"
        / "pinocchio"
        / "python"
        / "motion_matching"
        / "fit_swing_pinocchio.py",
        "import pinocchio_loaders\n",
    )

    findings = audit_engine_loaders(tmp_path)

    assert len(findings) == 1
    assert "pinocchio_loaders" in findings[0].reason


def test_audit_engine_loaders_ignores_tests_and_non_fit_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_fit_swing_opensim.py",
        "import opensim_loaders\n",
    )
    _write(
        tmp_path
        / "src"
        / "engines"
        / "physics_engines"
        / "opensim"
        / "python"
        / "motion_matching"
        / "load_fixture.py",
        "import opensim_loaders\n",
    )

    assert audit_engine_loaders(tmp_path) == []
