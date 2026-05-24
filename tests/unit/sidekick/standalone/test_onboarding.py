"""T8 TDD: StandaloneOnboarding sentinel logic and skip-onboarding flag.

Marker: headless_safe — no display or PyQt6 required.
Tests use tmp_path so nothing is written to the user's real ~/.config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sidekick.standalone.onboarding import (
    SENTINEL_FILENAME,
    OnboardingState,
    StandaloneOnboarding,
)


# ---------------------------------------------------------------------------
# T8-AC-2: sentinel controls whether onboarding runs
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_onboarding_needed_when_sentinel_absent(tmp_path: Path) -> None:
    ob = StandaloneOnboarding(config_dir=tmp_path)
    assert ob.needs_onboarding() is True


@pytest.mark.headless_safe
def test_onboarding_not_needed_when_sentinel_present(tmp_path: Path) -> None:
    (tmp_path / SENTINEL_FILENAME).touch()
    ob = StandaloneOnboarding(config_dir=tmp_path)
    assert ob.needs_onboarding() is False


@pytest.mark.headless_safe
def test_mark_complete_writes_sentinel(tmp_path: Path) -> None:
    ob = StandaloneOnboarding(config_dir=tmp_path)
    assert ob.needs_onboarding() is True
    ob.mark_complete()
    assert (tmp_path / SENTINEL_FILENAME).exists()
    assert ob.needs_onboarding() is False


# ---------------------------------------------------------------------------
# T8-AC-2: --skip-onboarding flag bypasses flow
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_skip_flag_bypasses_onboarding(tmp_path: Path) -> None:
    ob = StandaloneOnboarding(config_dir=tmp_path, skip=True)
    assert ob.needs_onboarding() is False


@pytest.mark.headless_safe
def test_skip_flag_does_not_write_sentinel(tmp_path: Path) -> None:
    ob = StandaloneOnboarding(config_dir=tmp_path, skip=True)
    _ = ob.needs_onboarding()
    assert not (tmp_path / SENTINEL_FILENAME).exists()


# ---------------------------------------------------------------------------
# T8-AC: state machine — 3 steps
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_onboarding_state_machine_sequence() -> None:
    states = list(OnboardingState)
    assert states[0] == OnboardingState.WELCOME
    assert states[1] == OnboardingState.PICK_PROFILE
    assert states[2] == OnboardingState.CONFIRM_DATA_DIR


@pytest.mark.headless_safe
def test_onboarding_advance_through_all_steps(tmp_path: Path) -> None:
    ob = StandaloneOnboarding(config_dir=tmp_path)
    assert ob.current_state() == OnboardingState.WELCOME

    ob.advance()
    assert ob.current_state() == OnboardingState.PICK_PROFILE

    ob.advance()
    assert ob.current_state() == OnboardingState.CONFIRM_DATA_DIR

    ob.advance()  # finish — writes sentinel
    assert ob.needs_onboarding() is False


@pytest.mark.headless_safe
def test_onboarding_is_complete_after_all_steps(tmp_path: Path) -> None:
    ob = StandaloneOnboarding(config_dir=tmp_path)
    for _ in range(len(OnboardingState)):
        ob.advance()
    assert ob.is_complete()


# ---------------------------------------------------------------------------
# T8-AC: DbC preconditions
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_config_dir_must_exist_or_be_creatable(tmp_path: Path) -> None:
    new_dir = tmp_path / "subdir"
    # Should not raise — StandaloneOnboarding must create the dir if needed
    ob = StandaloneOnboarding(config_dir=new_dir)
    ob.mark_complete()
    assert (new_dir / SENTINEL_FILENAME).exists()
