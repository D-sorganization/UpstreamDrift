"""State profile actions for the Sidekick sidebar."""

from __future__ import annotations

from pathlib import Path

from .state_profiles import SidekickStateProfileResult, SidekickStateProfileStore


class StateProfileMixin:
    """Mixin for profile persistence actions."""

    def save_state_profile(
        self,
        storage_root: str | Path,
        name: str,
    ) -> SidekickStateProfileResult:
        """Save the current sidebar state as a named Sidekick profile."""
        state = self.snapshot_state()
        result = SidekickStateProfileStore(storage_root).save_profile(name, state)
        if result.ok:
            self._state = state
        return result

    def load_state_profile(
        self,
        storage_root: str | Path,
        name: str,
    ) -> SidekickStateProfileResult:
        """Load and apply a named Sidekick profile atomically."""
        result = SidekickStateProfileStore(storage_root).load_profile(name)
        if result.ok and result.state is not None:
            self.apply_state(result.state)
        return result

    def clear_state_profiles(
        self,
        storage_root: str | Path,
        *,
        confirmation: str | None = None,
    ) -> SidekickStateProfileResult:
        """Clear stored Sidekick profiles after explicit confirmation."""
        return SidekickStateProfileStore(storage_root).clear_data(
            confirmation=confirmation
        )
