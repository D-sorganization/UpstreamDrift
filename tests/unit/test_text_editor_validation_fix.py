"""Tests for _text_editor_validation.py dead-code removal and robustness.

Covers:
- _find_element_line no longer calls ET.tostring (dead code removal)
- _find_element_line uses splitlines() for cross-platform compatibility
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # stdlib for Element/SubElement

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks
import pytest
from src.shared.python.model_generation.editor._text_editor_models import (
    ValidationMessage,
    ValidationSeverity,
)
from src.shared.python.model_generation.editor._text_editor_validation import (
    _URDFValidationMixin,
)


class TestFindElementLine:
    """Tests for the _find_element_line heuristic."""

    def test_finds_element_by_tag(self) -> None:
        """Should return the line number of a known element."""
        editor = _URDFValidationMixin()
        editor._content = (
            '<?xml version="1.0"?>\n'
            '<robot name="test">\n'
            '  <link name="base_link"/>\n'
            '  <link name="arm_link"/>\n'
            "</robot>\n"
        )
        root = DefusedET.fromstring(editor._content)
        link = root.find("link")
        assert link is not None
        line = editor._find_element_line(link)
        assert line == 3

    def test_finds_element_by_tag_and_name(self) -> None:
        """Should disambiguate elements with the same tag by name."""
        editor = _URDFValidationMixin()
        editor._content = (
            '<?xml version="1.0"?>\n'
            '<robot name="test">\n'
            '  <link name="base_link"/>\n'
            '  <link name="arm_link"/>\n'
            "</robot>\n"
        )
        root = DefusedET.fromstring(editor._content)
        links = root.findall("link")
        assert len(links) == 2
        line = editor._find_element_line(links[1])
        assert line == 4

    def test_returns_one_when_not_found(self) -> None:
        """Should return 1 when the element is not in content."""
        editor = _URDFValidationMixin()
        editor._content = "<robot/>"
        # Create an element that won't match
        elem = ET.Element("link")
        elem.set("name", "missing")
        line = editor._find_element_line(elem)
        assert line == 1

    def test_raises_on_none_element(self) -> None:
        """Precondition: elem must not be None."""
        editor = _URDFValidationMixin()
        editor._content = "<robot/>"
        with pytest.raises(ValueError, match="elem must be provided"):
            editor._find_element_line(None)  # type: ignore[arg-type]

    def test_splitlines_handles_crlf(self) -> None:
        """splitlines() should handle Windows-style line endings."""
        editor = _URDFValidationMixin()
        editor._content = (
            '<?xml version="1.0"?>\r\n'
            '<robot name="test">\r\n'
            '  <link name="base_link"/>\r\n'
            "</robot>\r\n"
        )
        root = DefusedET.fromstring(editor._content)
        link = root.find("link")
        assert link is not None
        line = editor._find_element_line(link)
        # Even with \r\n, splitlines() gives us correct line count
        assert line == 3

    def test_no_et_tostring_side_effects(self) -> None:
        """The method should not mutate the element or content."""
        editor = _URDFValidationMixin()
        original_content = (
            '<?xml version="1.0"?>\n'
            '<robot name="test">\n'
            '  <link name="base_link"/>\n'
            "</robot>\n"
        )
        editor._content = original_content
        root = DefusedET.fromstring(editor._content)
        link = root.find("link")
        assert link is not None
        _ = editor._find_element_line(link)
        assert editor._content == original_content
        assert link.get("name") == "base_link"


class TestURDFValidationMessages:
    """Sanity tests for validation message construction."""

    def test_validation_message_str_with_element(self) -> None:
        msg = ValidationMessage(
            severity=ValidationSeverity.ERROR,
            line=10,
            column=5,
            message="Test error",
            element="link",
        )
        assert str(msg) == "[ERROR] Line 10, Col 5 (link): Test error"

    def test_validation_message_str_without_element(self) -> None:
        msg = ValidationMessage(
            severity=ValidationSeverity.WARNING,
            line=1,
            column=0,
            message="Missing name",
        )
        assert str(msg) == "[WARNING] Line 1: Missing name"
