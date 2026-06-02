"""Regression tests for shared signal-toolkit imports.

The MuJoCo unified launcher (formerly exercised here) was retired in #7104;
the remaining tests guard the absolute-import path for shared modules.
"""


class TestSignalToolkitContractsImport:
    """Ensure signal_toolkit can import contracts via absolute path."""

    def test_contracts_require_importable(self) -> None:
        from src.shared.python.contracts import require

        assert callable(require)

    def test_signal_toolkit_core_importable(self) -> None:
        from src.shared.python.signal_toolkit.core import Signal

        assert Signal is not None
