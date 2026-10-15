"""CI guard: OpenSim-gated spine tests must fail, not skip, in the OpenSim lane (#12752).

When ``UD_REQUIRE_OPENSIM=1`` a spine test skipped only because ``opensim`` is
not importable is reported as a failure. Any other skip reason, and any run
without the variable, keeps the normal skip behaviour. The pure decision
logic is tested directly; the conftest hooks are exercised with ``pytester``
against an isolated copy of the real conftest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.opensim_lane_guard import (
    REQUIRE_ENV_VAR,
    failure_message,
    is_opensim_absence_skip,
    is_within_directory,
    opensim_lane_required,
    should_fail_skip,
)
from tests.helpers.opensim_support import SKIP_REASON

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.unit

CONFTEST = Path(__file__).resolve().parent / "conftest.py"
IMPORTORSKIP_REASON = "could not import 'opensim': No module named 'opensim'"
UNRELATED_REASON = "CUDA device not available"


def test_env_var_name_is_the_documented_lane_switch() -> None:
    assert REQUIRE_ENV_VAR == "UD_REQUIRE_OPENSIM"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_lane_required_for_truthy_values(value: str) -> None:
    assert opensim_lane_required({REQUIRE_ENV_VAR: value}) is True


@pytest.mark.parametrize("value", ["0", "", "false", "no", "off"])
def test_lane_not_required_for_falsy_values(value: str) -> None:
    assert opensim_lane_required({REQUIRE_ENV_VAR: value}) is False


def test_lane_not_required_when_variable_unset() -> None:
    assert opensim_lane_required({}) is False


def test_lane_required_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        opensim_lane_required(["UD_REQUIRE_OPENSIM=1"])  # type: ignore[arg-type]


def test_absence_detected_for_require_opensim_skip() -> None:
    assert is_opensim_absence_skip(SKIP_REASON) is True


def test_absence_detected_for_importorskip_message() -> None:
    assert is_opensim_absence_skip(IMPORTORSKIP_REASON) is True


def test_unrelated_skip_is_not_absence() -> None:
    assert is_opensim_absence_skip(UNRELATED_REASON) is False


def test_absence_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        is_opensim_absence_skip(None)  # type: ignore[arg-type]


def test_should_fail_only_when_lane_required_and_absent() -> None:
    lane = {REQUIRE_ENV_VAR: "1"}
    assert should_fail_skip(SKIP_REASON, lane) is True
    assert should_fail_skip(IMPORTORSKIP_REASON, lane) is True
    assert should_fail_skip(UNRELATED_REASON, lane) is False
    assert should_fail_skip(SKIP_REASON, {}) is False


def test_failure_message_names_env_var_and_original_reason() -> None:
    message = failure_message(SKIP_REASON)
    assert REQUIRE_ENV_VAR in message
    assert SKIP_REASON in message


def test_is_within_directory_accepts_descendant(tmp_path: Path) -> None:
    child = tmp_path / "spine" / "test_x.py"
    assert is_within_directory(child, tmp_path / "spine") is True


def test_is_within_directory_rejects_prefix_sibling(tmp_path: Path) -> None:
    sibling = tmp_path / "spine_other" / "test_x.py"
    assert is_within_directory(sibling, tmp_path / "spine") is False


def _inner_suite(pytester: pytest.Pytester, reason_call: str) -> None:
    """Build an isolated ``spine/`` package with the real conftest and one skip."""
    pytester.mkdir("spine")
    (pytester.path / "spine" / "conftest.py").write_text(
        CONFTEST.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (pytester.path / "spine" / "__init__.py").write_text("", encoding="utf-8")
    (pytester.path / "spine" / "test_inner.py").write_text(
        "import pytest\n\n\ndef test_native_spine():\n    " + reason_call + "\n",
        encoding="utf-8",
    )


def test_absent_opensim_skip_becomes_failure_in_lane(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(REQUIRE_ENV_VAR, "1")
    _inner_suite(pytester, f"pytest.skip({SKIP_REASON!r})")
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(failed=1, skipped=0)
    result.stdout.fnmatch_lines([f"*{REQUIRE_ENV_VAR}*"])


def test_importorskip_style_skip_becomes_failure_in_lane(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(REQUIRE_ENV_VAR, "1")
    _inner_suite(pytester, f"pytest.skip({IMPORTORSKIP_REASON!r})")
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(failed=1, skipped=0)


def test_absent_opensim_skip_stays_skipped_without_lane(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(REQUIRE_ENV_VAR, raising=False)
    _inner_suite(pytester, f"pytest.skip({SKIP_REASON!r})")
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(failed=0, skipped=1)


def test_unrelated_skip_stays_skipped_in_lane(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(REQUIRE_ENV_VAR, "1")
    _inner_suite(pytester, f"pytest.skip({UNRELATED_REASON!r})")
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(failed=0, skipped=1)
