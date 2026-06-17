# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the offline GUI help center."""

from __future__ import annotations

from movement_optimizer.gui.help_dialog import (
    GLOSSARY,
    HELP_TOPICS,
    HelpCenterDialog,
    ParameterHelpDialog,
)


def test_help_center_exposes_required_offline_topics(qapp) -> None:
    dialog = HelpCenterDialog()

    assert len(HELP_TOPICS) >= 5
    assert dialog.tabs.count() >= 5
    assert {"getting_started", "parameters", "results", "troubleshooting", "glossary"} <= set(
        HELP_TOPICS
    )


def test_help_center_can_select_each_topic(qapp) -> None:
    dialog = HelpCenterDialog(initial_topic="getting_started")

    for topic_id, topic in HELP_TOPICS.items():
        dialog.select_topic(topic_id)
        assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == topic.title


def test_help_center_contains_glossary_terms(qapp) -> None:
    dialog = HelpCenterDialog(initial_topic="glossary")

    assert len(GLOSSARY) >= 7
    assert {"COM", "BOS", "Torque", "ROM"} <= set(GLOSSARY)
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == HELP_TOPICS["glossary"].title


def test_parameter_help_dialog_opens_parameter_topic(qapp) -> None:
    dialog = ParameterHelpDialog()

    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == HELP_TOPICS["parameters"].title
