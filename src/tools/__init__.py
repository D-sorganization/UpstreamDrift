"""Tools package for Golf Modeling Suite.

Contains utilities for URDF generation, MATLAB integration, and other tools.
"""

from .check_markdown_links import (
    check_links,
    extract_links_from_markdown,
    resolve_and_verify_link,
)
from .code_quality_check import (
    Colors,
    check_ast_issues,
    check_banned_patterns,
    check_file,
    check_magic_numbers,
    is_legitimate_pass_context,
    main,
)

__all__: list[str] = [
    # check_markdown_links
    "check_links",
    "extract_links_from_markdown",
    "resolve_and_verify_link",
    # code_quality_check
    "Colors",
    "check_ast_issues",
    "check_banned_patterns",
    "check_file",
    "check_magic_numbers",
    "is_legitimate_pass_context",
    "main",
]


# Dummy comment for PR 6236
