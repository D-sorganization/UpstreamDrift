"""Unit tests for the cmeel native-runtime bootstrap helper (#9607).

Covers pure logic only: cmeel prefix resolution, LD_LIBRARY_PATH assembly,
and the DbC failure diagnostics. The Linux ``import pinocchio`` behavior
itself is validated by the ``articulated-manufactured-authority`` CI job.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


from scripts.research.proximal_distal_energy.articulated_native_runtime import (
    REQUIRED_CMEEL_SONAMES,
    assemble_ld_library_path,
    require_cmeel_sonames,
    resolve_cmeel_prefix_lib,
)


def _make_cmeel_prefix(site_packages: Path, sonames: tuple[str, ...] = ()) -> Path:
    lib_dir = site_packages / "cmeel.prefix" / "lib"
    lib_dir.mkdir(parents=True)
    for soname in sonames:
        (lib_dir / soname).write_bytes(b"")
    return lib_dir


def test_resolve_cmeel_prefix_lib_returns_first_candidate_with_prefix(
    tmp_path: Path,
) -> None:
    bare = tmp_path / "bare-site-packages"
    bare.mkdir()
    prefixed = tmp_path / "venv-site-packages"
    expected = _make_cmeel_prefix(prefixed, ("liburdfdom_sensor.so.4.0",))

    assert resolve_cmeel_prefix_lib([bare, prefixed]) == expected


def test_resolve_cmeel_prefix_lib_fails_listing_every_candidate(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    with pytest.raises(RuntimeError) as error:
        resolve_cmeel_prefix_lib([first, second])
    message = str(error.value)
    assert "cmeel.prefix/lib" in message
    assert str(first) in message
    assert str(second) in message


def test_require_cmeel_sonames_passes_when_every_soname_is_present(
    tmp_path: Path,
) -> None:
    lib_dir = _make_cmeel_prefix(tmp_path / "site-packages", REQUIRED_CMEEL_SONAMES)

    require_cmeel_sonames(lib_dir)


def test_require_cmeel_sonames_names_missing_soname_and_directory(
    tmp_path: Path,
) -> None:
    lib_dir = _make_cmeel_prefix(
        tmp_path / "site-packages",
        ("liburdfdom_model.so.4.0",),
    )

    with pytest.raises(RuntimeError) as error:
        require_cmeel_sonames(lib_dir)
    message = str(error.value)
    assert "liburdfdom_sensor.so.4.0" in message
    assert str(lib_dir) in message


def test_assemble_ld_library_path_prepends_and_deduplicates(tmp_path: Path) -> None:
    lib_dir = tmp_path / "cmeel.prefix" / "lib"

    separator = os.pathsep
    assert (
        assemble_ld_library_path(lib_dir, "/opt/existing")
        == f"{lib_dir}{separator}/opt/existing"
    )
    assert assemble_ld_library_path(lib_dir, None) == str(lib_dir)
    assert (
        assemble_ld_library_path(lib_dir, f"{lib_dir}{separator}/opt/existing")
        == f"{lib_dir}{separator}/opt/existing"
    )


def test_ensure_cmeel_native_library_path_reexecs_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-exec once with the prefix prepended, then no-op after re-exec."""

    import scripts.research.proximal_distal_energy.articulated_native_runtime as runtime

    lib_dir = _make_cmeel_prefix(tmp_path / "site-packages", REQUIRED_CMEEL_SONAMES)
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(
        runtime.sys,
        "argv",
        ["/venv/bin/python", "--profile", "authority"],
    )
    monkeypatch.setattr(
        runtime, "_candidate_site_packages", lambda: [tmp_path / "site-packages"]
    )
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    executed: list[list[str]] = []
    monkeypatch.setattr(runtime.os, "execv", lambda exe, argv: executed.append(argv))

    runtime.ensure_cmeel_native_library_path(
        module_name="scripts.research.proximal_distal_energy.module"
    )
    assert executed == [
        [
            runtime.sys.executable,
            "-m",
            "scripts.research.proximal_distal_energy.module",
            "--profile",
            "authority",
        ]
    ]
    assert os.environ["LD_LIBRARY_PATH"] == str(lib_dir)

    runtime.ensure_cmeel_native_library_path(
        module_name="scripts.research.proximal_distal_energy.module"
    )
    assert len(executed) == 1
