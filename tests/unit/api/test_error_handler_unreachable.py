"""Error-handler contract tests (issue #7167 D3).

The decorator's post-handler ``return None`` was unreachable but, if the
invariant ever broke (a new exception type added to the catch tuple without a
branch in ``_handle_common_exceptions``), it would hand a body-less None to the
framework. It is now an executable assertion.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.middleware import error_handler
from src.api.middleware.error_handler import (
    _handle_common_exceptions,
    handle_api_errors,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ValueError("bad"), 400),
        (FileNotFoundError("missing"), 404),
        (PermissionError("denied"), 403),
        (NotImplementedError("nope"), 501),
        (RuntimeError("boom"), 500),
        (KeyError("k"), 500),
    ],
)
def test_handle_common_exceptions_always_raises_http(
    exc: Exception, status: int
) -> None:
    with pytest.raises(HTTPException) as info:
        _handle_common_exceptions(exc, "fn")
    assert info.value.status_code == status


def test_sync_decorator_surfaces_http_exception_not_none() -> None:
    @handle_api_errors
    def boom() -> str:
        raise ValueError("invalid input")

    with pytest.raises(HTTPException) as info:
        boom()
    assert info.value.status_code == 400


def test_decorator_raises_assertion_if_handler_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate the broken invariant: _handle_common_exceptions returns instead
    # of raising. The decorator must fail loudly, never return None.
    monkeypatch.setattr(
        error_handler, "_handle_common_exceptions", lambda e, name: None
    )

    @handle_api_errors
    def boom() -> str:
        raise ValueError("invalid input")

    with pytest.raises(AssertionError, match="unreachable"):
        boom()
