import os
import sys

from PyQt6.QtWidgets import QApplication

# Mocking parts of the system if needed, but since I have the shared library it should work.
# We need to make sure PYTHONPATH includes src/

getcwd = os.getcwd
sys_path = sys.path
sys_path.append(getcwd())


def verify_launcher_init() -> bool | None:
    QApplication(sys.argv)

    # Try to import and init the launcher
    try:
        from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

        # We don't want to actually show it or start the event loop
        launcher = UpstreamDriftLauncher()

        # Check if tabs exist
        if hasattr(launcher, "main_tabs"):
            main_tabs = launcher.main_tabs
            assert main_tabs is not None, "main_tabs must not be None"
            tab_count = main_tabs.count()
            for i in range(tab_count):
                main_tabs.tabText(i)
        else:
            return False

        if hasattr(launcher, "data_processor"):
            pass
        else:
            return False

        return True
    except Exception as e:  # noqa: BLE001, F841
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if verify_launcher_init():
        sys.exit(0)
    else:
        sys.exit(1)
