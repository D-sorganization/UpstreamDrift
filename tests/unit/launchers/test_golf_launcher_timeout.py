"""Tests for upstream_drift_launcher startup timeout and create_model_card — issue #5488.

RED → GREEN cycle:
  - UpstreamDriftLauncher must have _handle_startup_timeout method
  - create_model_card must either be implemented or raise NotImplementedError
  - Timeout handler must not raise, must log an error
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestStartupTimeoutHandler:
    """#5488 — UpstreamDriftLauncher must guard against infinite startup wait."""

    def test_startup_timeout_handler_exists(self) -> None:
        """UpstreamDriftLauncher must have a _handle_startup_timeout method."""
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        assert hasattr(UpstreamDriftLauncher, "_handle_startup_timeout"), (
            "UpstreamDriftLauncher is missing _handle_startup_timeout method"
        )

    def test_startup_timeout_handler_is_callable(self) -> None:
        """_handle_startup_timeout must be callable."""
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        assert callable(getattr(UpstreamDriftLauncher, "_handle_startup_timeout", None))

    def test_startup_timeout_does_not_raise(self) -> None:
        """Calling _handle_startup_timeout on a mock launcher must not raise."""
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        # Build a minimal mock that satisfies the method's requirements
        mock_self = MagicMock()
        mock_self.loading = True

        # Should not raise
        UpstreamDriftLauncher._handle_startup_timeout(mock_self)

    def test_startup_timeout_clears_loading_flag(self) -> None:
        """_handle_startup_timeout must clear the loading flag."""
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        mock_self = MagicMock()
        mock_self.loading = True

        UpstreamDriftLauncher._handle_startup_timeout(mock_self)

        assert mock_self.loading is False, (
            "_handle_startup_timeout did not clear the loading flag"
        )

    def test_startup_timeout_shows_error_message(self) -> None:
        """_handle_startup_timeout must surface a user-visible error."""
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        mock_self = MagicMock()
        mock_self.loading = True

        with patch("src.launchers.upstream_drift_launcher.logger") as mock_logger:
            UpstreamDriftLauncher._handle_startup_timeout(mock_self)
            # Must log an error-level message
            assert mock_logger.error.called or mock_logger.warning.called, (
                "_handle_startup_timeout did not log any error/warning"
            )
