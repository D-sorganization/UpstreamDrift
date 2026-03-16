import os
import sys

from PyQt6.QtWidgets import QApplication

# Mocking parts of the system if needed, but since I have the shared library it should work.
# We need to make sure PYTHONPATH includes src/

getcwd = os.getcwd
sys_path = sys.path
sys_path.append(getcwd())


def verify_launcher_init():
    QApplication(sys.argv)

    # Try to import and init the launcher
    try:
        from src.launchers.golf_launcher import GolfLauncher

        # We don't want to actually show it or start the event loop
        launcher = GolfLauncher()
        print("Launcher initialized successfully")

        # Check if tabs exist
        if hasattr(launcher, "main_tabs"):
            main_tabs = launcher.main_tabs
            tab_count = main_tabs.count()
            print(f"Tabs found: {tab_count}")
            for i in range(tab_count):
                print(f"  Tab {i}: {main_tabs.tabText(i)}")
        else:
            print("ERROR: main_tabs not found")
            return False

        if hasattr(launcher, "data_processor"):
            print("DataProcessorWidget integrated successfully")
        else:
            print("ERROR: data_processor not found")
            return False

        return True
    except Exception as e:
        print(f"FAILED to initialize launcher: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if verify_launcher_init():
        sys.exit(0)
    else:
        sys.exit(1)
