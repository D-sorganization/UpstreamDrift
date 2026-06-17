# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Thin wrapper around Qt's translation system.

Usage:
    from movement_optimizer.i18n import tr
    label = QLabel(tr("Body Mass"))
"""

from __future__ import annotations

import logging

_translator = None
_logger = logging.getLogger(__name__)


def setup_translations(locale: str | None = None) -> None:
    """Install Qt translator for the given locale (e.g. 'de', 'ja').

    Falls back to English if translation not available.
    """
    import os

    from PyQt6.QtCore import QLocale, QTranslator
    from PyQt6.QtWidgets import QApplication

    global _translator
    app = QApplication.instance()
    if app is None:
        return

    # Remove any previously installed translator so language switches are clean.
    if _translator is not None:
        app.removeTranslator(_translator)
        _translator = None

    if locale is None:
        locale = QLocale.system().name()  # e.g. 'en_US', 'de_DE'

    _translator = QTranslator()
    locale_dir = os.path.join(os.path.dirname(__file__), "locale")
    loaded = _translator.load(f"movement_optimizer_{locale}", locale_dir)
    if loaded:
        app.installTranslator(_translator)
        _logger.info("Loaded translations for locale: %s", locale)
    else:
        _logger.debug("No translation found for locale: %s, using English", locale)


def tr(text: str) -> str:
    """Mark string for translation and return translated version."""
    from PyQt6.QtCore import QCoreApplication

    return QCoreApplication.translate("movement_optimizer", text)
