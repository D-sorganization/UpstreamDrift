"""Unit tests for src/tools/check_markdown_links.py."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tools import check_markdown_links as cml


class TestExtractLinks:
    def test_extracts_relative_links(self) -> None:
        content = "See [doc](./foo.md) and [bar](sub/bar.md)."
        assert cml.extract_links_from_markdown(content) == ["./foo.md", "sub/bar.md"]

    def test_skips_external_and_anchor_links(self) -> None:
        content = (
            "[a](http://example.com) [b](https://x.org) [c](mailto:x@y.z) [d](#section)"
        )
        assert cml.extract_links_from_markdown(content) == []

    def test_mixed_links(self) -> None:
        content = "[ext](https://x) [rel](file.md) [anchor](#a)"
        assert cml.extract_links_from_markdown(content) == ["file.md"]

    def test_empty_content(self) -> None:
        assert cml.extract_links_from_markdown("") == []

    def test_no_links(self) -> None:
        assert cml.extract_links_from_markdown("Hello world.") == []

    def test_requires_string(self) -> None:
        with pytest.raises(Exception):  # noqa: BLE001, PT011, B017
            cml.extract_links_from_markdown(None)  # type: ignore[arg-type]


class TestResolveAndVerify:
    def test_existing_relative(self, tmp_path: Path) -> None:
        target = tmp_path / "real.md"
        target.write_text("x", encoding="utf-8")
        assert cml.resolve_and_verify_link("real.md", tmp_path) is None

    def test_broken_link_returns_message(self, tmp_path: Path) -> None:
        err = cml.resolve_and_verify_link("missing.md", tmp_path)
        assert err is not None
        assert "Broken link" in err

    def test_url_encoded_link(self, tmp_path: Path) -> None:
        target = tmp_path / "my file.md"
        target.write_text("x", encoding="utf-8")
        assert cml.resolve_and_verify_link("my%20file.md", tmp_path) is None

    def test_anchor_only_returns_none(self, tmp_path: Path) -> None:
        # link split on '#' leaves empty path
        assert cml.resolve_and_verify_link("#top", tmp_path) is None

    def test_link_with_anchor_resolves_to_file(self, tmp_path: Path) -> None:
        target = tmp_path / "foo.md"
        target.write_text("x", encoding="utf-8")
        assert cml.resolve_and_verify_link("foo.md#section", tmp_path) is None

    def test_link_with_anchor_broken(self, tmp_path: Path) -> None:
        result = cml.resolve_and_verify_link("nope.md#section", tmp_path)
        assert result is not None
        assert "Broken link" in result

    def test_requires_path(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):  # noqa: BLE001, PT011, B017
            cml.resolve_and_verify_link("x", "not a path")  # type: ignore[arg-type]

    def test_oserror_returns_message(self, tmp_path: Path) -> None:
        # Force an OSError during resolve
        with patch.object(Path, "resolve", side_effect=OSError("boom")):
            err = cml.resolve_and_verify_link("a.md", tmp_path)
        assert err is not None
        assert "Invalid path configuration" in err


class TestCheckLinks:
    def test_clean_tree(self, tmp_path: Path) -> None:
        (tmp_path / "real.md").write_text("x")
        (tmp_path / "doc.md").write_text("[ok](real.md)")
        assert cml.check_links(tmp_path) == []

    def test_broken_link_reported(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text("[broken](missing.md)")
        errors = cml.check_links(tmp_path)
        assert len(errors) == 1
        assert "Broken link" in errors[0]

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        nm = tmp_path / "node_modules" / "lib"
        nm.mkdir(parents=True)
        (nm / "x.md").write_text("[broken](missing.md)")
        assert cml.check_links(tmp_path) == []

    def test_skips_git(self, tmp_path: Path) -> None:
        git = tmp_path / ".git"
        git.mkdir()
        (git / "x.md").write_text("[broken](missing.md)")
        assert cml.check_links(tmp_path) == []

    def test_unreadable_file_recorded(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text("[ok](real.md)")
        (tmp_path / "real.md").write_text("x")
        with patch.object(Path, "read_text", side_effect=PermissionError("nope")):
            errors = cml.check_links(tmp_path)
        assert any("Could not read" in e for e in errors)

    def test_requires_existing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):  # noqa: BLE001, PT011, B017
            cml.check_links(tmp_path / "nope")

    def test_requires_path_type(self) -> None:
        with pytest.raises(Exception):  # noqa: BLE001, PT011, B017
            cml.check_links("not a path")  # type: ignore[arg-type]


class TestMain:
    def test_main_exits_zero_no_errors(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as ei:
            cml.main([])
        assert ei.value.code == 0

    @pytest.mark.unit
    def test_main_exits_nonzero_with_errors(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """A broken link must fail the process so CI can gate on it (#8851)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "doc.md").write_text("[bad](missing.md)")
        caplog.set_level(logging.WARNING, logger=cml.logger.name)
        with pytest.raises(SystemExit) as ei:
            cml.main([])
        assert ei.value.code == 1
        assert any("Broken link" in r.message for r in caplog.records)

    @pytest.mark.unit
    def test_main_scans_explicit_paths(self, tmp_path: Path, monkeypatch) -> None:
        """Explicit paths override the default documentation roots."""
        monkeypatch.chdir(tmp_path)
        clean = tmp_path / "clean"
        clean.mkdir()
        (clean / "ok.md").write_text("[self](ok.md)")
        (tmp_path / "broken.md").write_text("[bad](missing.md)")
        with pytest.raises(SystemExit) as ei:
            cml.main(["clean"])
        assert ei.value.code == 0

    @pytest.mark.unit
    def test_default_roots_include_docs_help(self, tmp_path: Path) -> None:
        """docs/help/ is in scope by default (#9413)."""
        help_dir = tmp_path / "docs" / "help"
        help_dir.mkdir(parents=True)
        (tmp_path / "README.md").write_text("# root")
        roots = cml.default_roots(tmp_path)
        assert help_dir in roots
        assert tmp_path / "docs" in roots
        assert tmp_path / "README.md" in roots

    @pytest.mark.unit
    def test_exempt_link_prefix_is_skipped(self, tmp_path: Path) -> None:
        """Uninitialised submodule targets are not reported as broken."""
        (tmp_path / "doc.md").write_text("[vendored](vendor/ud-tools/thing.md)")
        assert cml.check_links(tmp_path) == []

    @pytest.mark.unit
    def test_placeholder_link_is_skipped(self, tmp_path: Path) -> None:
        """Documentation placeholders are not treated as paths."""
        assert cml.extract_links_from_markdown("[x](<pyproject URL>)") == []
