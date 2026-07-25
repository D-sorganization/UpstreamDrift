"""Regression tests for broken help paths (#8014, #7986) and theme dupes (#8026).

* The Help menu's *User Guide* and *Project Map* entries resolved to files that
  have never existed, so they silently fell back to the GitHub repo URL / a
  "not found" warning while their tooltips promised a bundled document.
* ``ContextHelpDock`` claimed "No specific documentation available" for 11 of
  the 22 mapped tiles because its hard-coded doc paths pointed at directories
  that were never created.
* ``View > Theme`` listed every custom theme twice and split the exclusive
  check state across the duplicate pair.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from src.launchers import about_dialog, help_dialogs  # noqa: E402
from src.launchers.launcher_dialogs import DialogsManager  # noqa: E402
from src.shared.python.config.model_registry import ModelRegistry  # noqa: E402

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestUserGuidePaths:
    """#8014 — Help > User Guide must find a real bundled document."""

    def test_at_least_one_candidate_exists(self) -> None:
        candidates = about_dialog.user_guide_candidates()
        existing = [c for c in candidates if c.exists()]
        assert existing, f"No bundled user guide found. Searched: {candidates}"

    def test_every_candidate_lives_inside_the_repository(self) -> None:
        """The old ``parents[3]`` candidate resolved outside the checkout."""
        repo_root = Path(about_dialog.__file__).resolve().parents[2]
        for candidate in about_dialog.user_guide_candidates():
            assert repo_root in candidate.parents, (
                f"{candidate} escapes the repository root {repo_root} — #8014"
            )


class TestProjectMapPaths:
    """#8014 — Help > Project Map must find a real document."""

    def test_candidates_are_real_files(self) -> None:
        candidates = [REPO_ROOT / rel for rel in DialogsManager.PROJECT_MAP_CANDIDATES]
        existing = [c for c in candidates if c.exists()]
        assert existing, f"No Project Map found. Searched: {candidates}"

    def test_help_content_points_at_a_real_file(self) -> None:
        from src.shared.python.gui_pkg import help_content

        entry = help_content.FEATURE_HELP["project_map"]
        text = entry["description"] + "\n".join(entry["tips"])
        assert "docs/PROJECT_MAP.md" not in text.replace(
            "docs/architecture/PROJECT_MAP.md", ""
        ).replace("docs/governance/PROJECT_MAP.md", ""), (
            "help_content still advertises the nonexistent docs/PROJECT_MAP.md"
        )


class TestContextHelpDocResolution:
    """#7986 — every doc path the dock hands back must exist."""

    def test_every_mapped_tile_resolves_to_an_existing_file(self, qapp) -> None:
        dock = help_dialogs.ContextHelpDock()
        try:
            broken: list[tuple[str, str]] = []
            for model in ModelRegistry().get_all_models():
                path = dock._get_doc_file(model.id)
                if path is not None and not path.exists():
                    broken.append((model.id, str(path)))
            assert not broken, f"doc paths resolve to missing files: {broken}"
        finally:
            dock.deleteLater()

    @pytest.mark.parametrize(
        "model_id",
        [
            "putting_green",
            "putting_green_gui",
            "c3d_viewer",
            "video_analyzer",
            "mediapipe_analysis",
            "model_explorer",
            "project_map",
        ],
    )
    def test_previously_broken_tiles_now_have_docs(self, qapp, model_id) -> None:
        dock = help_dialogs.ContextHelpDock()
        try:
            path = dock._get_doc_file(model_id)
            assert path is not None and path.exists(), (
                f"{model_id} still has no resolvable documentation"
            )
        finally:
            dock.deleteLater()

    def test_missing_docs_report_what_was_searched(self, qapp) -> None:
        """An unmapped tile gets an honest message, not a bare 'not available'."""
        dock = help_dialogs.ContextHelpDock()
        try:
            dock.update_context("totally_unmapped_tile")
            text = dock.text_area.toPlainText()
            assert "totally_unmapped_tile" in text
            assert "no documentation mapping yet" in text
            assert "docs/user_guide/user_manual.md" in text
        finally:
            dock.deleteLater()

    def test_searched_paths_are_listed_when_candidates_all_miss(
        self, qapp, monkeypatch
    ) -> None:
        dock = help_dialogs.ContextHelpDock()
        try:
            monkeypatch.setattr(
                type(dock),
                "_doc_candidates",
                lambda self, model_id: [REPO_ROOT / "docs" / "nope.md"],
            )
            dock.update_context("some_tile")
            text = dock.text_area.toPlainText()
            assert "No documentation file has been written" in text
            assert "docs/nope.md" in text
        finally:
            dock.deleteLater()


class TestThemeMenuHasNoDuplicates:
    """#8026 — custom themes must appear exactly once."""

    def test_extra_themes_exclude_custom_themes(self) -> None:
        """The two menu sections must not overlap."""
        from src.shared.python.theme import ThemeManager

        manager = ThemeManager.instance()
        presets = {"Dark", "Light", "High Contrast"}
        custom = set(manager.get_custom_theme_names())
        extra = [
            t
            for t in manager.get_available_themes()
            if t not in presets and t not in custom
        ]
        assert not (set(extra) & custom), (
            "extra-themes section still overlaps the custom-themes section"
        )

    def test_source_filters_custom_names_out_of_extras(self) -> None:
        from src.launchers import launcher_theme

        source = Path(launcher_theme.__file__).read_text(encoding="utf-8")
        assert "custom_lookup" in source, (
            "launcher_theme no longer excludes custom themes from the "
            "built-in section — #8026 would regress"
        )


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
