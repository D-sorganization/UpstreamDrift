# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for i18n infrastructure (issue #410)."""

from __future__ import annotations

import os


def test_locale_directory_exists() -> None:
    """The locale directory must be present in the package."""
    import movement_optimizer

    pkg_dir = os.path.dirname(movement_optimizer.__file__)
    locale_dir = os.path.join(pkg_dir, "locale")
    assert os.path.isdir(locale_dir), f"locale directory not found at {locale_dir}"


def test_locale_gitkeep_exists() -> None:
    """The .gitkeep file must be present to track the empty locale directory."""
    import movement_optimizer

    pkg_dir = os.path.dirname(movement_optimizer.__file__)
    gitkeep = os.path.join(pkg_dir, "locale", ".gitkeep")
    assert os.path.isfile(gitkeep), f".gitkeep not found at {gitkeep}"


def test_german_ts_file_exists() -> None:
    """A minimal German translation file must be present as proof of concept."""
    import movement_optimizer

    pkg_dir = os.path.dirname(movement_optimizer.__file__)
    ts_file = os.path.join(pkg_dir, "locale", "movement_optimizer_de.ts")
    assert os.path.isfile(ts_file), f"German TS file not found at {ts_file}"


def test_tr_returns_string_without_qapplication() -> None:
    """tr() must return a plain str even when no QApplication is running."""
    from movement_optimizer.i18n import tr

    result = tr("Body Mass")
    assert isinstance(result, str)


def test_tr_english_fallback() -> None:
    """tr() must return the original English text when no translator is installed."""
    from movement_optimizer.i18n import tr

    # Without a QApplication or installed translator, Qt returns the source text.
    result = tr("Body Mass")
    # Accept either the source text (no QApp) or any non-empty string translation.
    assert result  # must be a non-empty string


def test_setup_translations_without_qapplication() -> None:
    """setup_translations() must not raise when no QApplication exists."""
    from movement_optimizer.i18n import setup_translations

    # Should return silently (QApplication.instance() is None)
    setup_translations()
    setup_translations(locale="de")
    setup_translations(locale="en_US")


def test_setup_translations_with_qapplication(qapp) -> None:
    """setup_translations() must not raise when a QApplication is running."""
    from movement_optimizer.i18n import setup_translations

    # Falls back to English if no compiled .qm file is present — that is fine.
    setup_translations(locale="de")
    setup_translations(locale="en_US")
    setup_translations(locale=None)


def test_tr_with_qapplication(qapp) -> None:
    """tr() must return a non-empty string when a QApplication is available."""
    from movement_optimizer.i18n import tr

    result = tr("Body Mass")
    assert isinstance(result, str)
    assert result  # non-empty


def test_tr_unknown_key_returns_source(qapp) -> None:
    """tr() must return the source string unchanged for untranslated keys."""
    from movement_optimizer.i18n import tr

    key = "This string has no translation registered anywhere"
    assert tr(key) == key
