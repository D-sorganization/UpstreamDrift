"""Standalone Sidekick first-run onboarding state machine.

Tracks the three-step first-run flow (Welcome → Pick Profile → Confirm Data Dir)
and persists a sentinel file so the flow runs exactly once.

GUI rendering is the caller's responsibility — this module is intentionally
free of PyQt6 imports so it can be exercised in headless CI tests.

Sentinel: ``~/.config/sidekick/onboarded`` (or the config_dir arg in tests).
"""

from __future__ import annotations

import enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SENTINEL_FILENAME = "onboarded"

# Default config directory (overridden in tests via constructor arg)
try:
    import platformdirs

    _DEFAULT_CONFIG_DIR = Path(platformdirs.user_config_dir("sidekick"))
except ImportError:  # pragma: no cover
    _DEFAULT_CONFIG_DIR = Path.home() / ".config" / "sidekick"


class OnboardingState(enum.Enum):
    """Three-step onboarding state machine.

    Invariant: states iterate in the order defined below.
    """

    WELCOME = "welcome"
    PICK_PROFILE = "pick_profile"
    CONFIRM_DATA_DIR = "confirm_data_dir"


_STATES: list[OnboardingState] = list(OnboardingState)


class StandaloneOnboarding:
    """Manages the first-run onboarding sentinel and step progression.

    Args:
        config_dir: Directory where the sentinel file is written.
                    Created automatically if absent.
        skip:       When True, ``needs_onboarding()`` always returns False
                    without touching the filesystem.  Corresponds to the
                    ``--skip-onboarding`` CLI flag.

    Precondition:
        ``config_dir`` must be a ``Path`` (or ``None`` to use the platform
        default).
    """

    def __init__(
        self,
        config_dir: Path | None = None,
        skip: bool = False,
    ) -> None:
        self._config_dir: Path = (
            Path(config_dir) if config_dir is not None else _DEFAULT_CONFIG_DIR
        )
        self._skip = skip
        self._step_index: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Sentinel checks
    # ------------------------------------------------------------------

    def _sentinel(self) -> Path:
        return self._config_dir / SENTINEL_FILENAME

    def needs_onboarding(self) -> bool:
        """Return True when onboarding has not been completed and skip is off.

        Postcondition: returns False when either the sentinel exists or
                       ``skip=True`` was passed at construction time.
        """
        if self._skip:
            return False
        if self._done:
            return False
        return not self._sentinel().exists()

    def mark_complete(self) -> None:
        """Write the sentinel and mark the state machine as finished.

        Creates the config directory if it does not exist.

        Postcondition: ``needs_onboarding()`` returns False after this call.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._sentinel().touch(exist_ok=True)
        self._done = True

        assert not self.needs_onboarding(), "postcondition: onboarding marked complete"

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def current_state(self) -> OnboardingState:
        """Return the current onboarding step.

        Precondition: state machine has not been exhausted.
        """
        assert self._step_index < len(_STATES), "onboarding already complete"
        return _STATES[self._step_index]

    def advance(self) -> None:
        """Move to the next step; write sentinel when the last step passes.

        Postcondition: if this was the last step, ``needs_onboarding()`` is False.
        """
        self._step_index += 1
        if self._step_index >= len(_STATES):
            self.mark_complete()

    def is_complete(self) -> bool:
        """Return True when all steps have been advanced through."""
        return self._done or self._step_index >= len(_STATES)

    # ------------------------------------------------------------------
    # Convenience: collect profile choice during PICK_PROFILE step
    # ------------------------------------------------------------------

    def collect_profile(self, profile: str) -> None:
        """Record the chosen profile.  No-op if skipping.

        Args:
            profile: ``"chat-first"`` or ``"calc-first"``.

        Raises:
            ValueError: if *profile* is not a known profile.
        """
        from sidekick.standalone.preferences import VALID_PROFILES

        if profile not in VALID_PROFILES:
            raise ValueError(
                f"Unknown profile {profile!r}; valid: {sorted(VALID_PROFILES)}"
            )
        self._chosen_profile: str = profile

    def chosen_profile(self) -> str | None:
        """Return the profile chosen during onboarding, or None if not yet chosen."""
        return getattr(self, "_chosen_profile", None)
