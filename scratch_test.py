import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock dependencies
pkg_mocks = {
    "mujoco": MagicMock(),
    "scipy": MagicMock(),
    "scipy.spatial": MagicMock(),
    "scipy.optimize": MagicMock(),
    "scipy.interpolate": MagicMock(),
    "scipy.signal": MagicMock(),
    "scipy.linalg": MagicMock(),
    "scipy.spatial.transform": MagicMock(),
    "matplotlib": MagicMock(),
    "matplotlib.pyplot": MagicMock(),
    "matplotlib.animation": MagicMock(),
    "matplotlib.figure": MagicMock(),
    "matplotlib.backends": MagicMock(),
    "matplotlib.backends.backend_qtagg": MagicMock(),
    "pinocchio": MagicMock(),
    "PyQt6": MagicMock(),
    "PyQt6.QtWidgets": MagicMock(),
    "PyQt6.QtCore": MagicMock(),
    "PyQt6.QtGui": MagicMock(),
}

with patch.dict(sys.modules, pkg_mocks):
    import recording_library

def test_get_unique_values():
    lib_path = Path("scratch/test_lib")
    lib_path.mkdir(parents=True, exist_ok=True)
    lib = recording_library.RecordingLibrary(str(lib_path))
    
    # Add some data
    conn = sqlite3.connect(str(lib.db_path))
    cursor = conn.cursor()
    cursor.execute("INSERT INTO recordings (golfer_name, club_type, filename) VALUES (?, ?, ?)", ("Alice", "Driver", "a.json"))
    cursor.execute("INSERT INTO recordings (golfer_name, club_type, filename) VALUES (?, ?, ?)", ("Bob", "Putter", "b.json"))
    cursor.execute("INSERT INTO recordings (golfer_name, club_type, filename) VALUES (?, ?, ?)", ("Alice", "Iron", "c.json"))
    conn.commit()
    conn.close()
    
    # Test get_unique_values
    golfers = lib.get_unique_values("golfer_name")
    assert golfers == ["Alice", "Bob"]
    
    clubs = lib.get_unique_values("club_type")
    assert clubs == ["Driver", "Iron", "Putter"]
    
    # Test invalid field
    try:
        lib.get_unique_values("invalid_field")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid field" in str(e)

    print("Test get_unique_values passed!")

if __name__ == "__main__":
    test_get_unique_values()
