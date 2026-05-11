import json
from unittest.mock import MagicMock, patch

import numpy as np

from src.shared.python.motion_matching.body_target import BodyTarget
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.multi_source_target import MultiSourceTarget
from simscape_adapter import SimscapeAdapter


def test_compute_cost_with_body_target_serializes_json():
    # Setup mock adapter without starting engine
    adapter = SimscapeAdapter()
    
    mock_eng = MagicMock()
    mock_eng.default_cost_options.return_value = {"w_position": 1.0}
    mock_eng.compute_cost.return_value = (42.0, {})
    
    # We mock the engine property to avoid matlab.engine import issues
    adapter._engine = mock_eng
    
    from src.shared.python.motion_matching.club_target import SourceProvenance
    club = ClubTarget(
        time=np.array([0.0, 0.1]),
        butt=np.zeros((2, 3)),
        clubhead=np.zeros((2, 3)),
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        impact_idx=1,
        source=SourceProvenance("dummy.xlsx", "excel", "TW", "swing", "0000")
    )
    
    body = BodyTarget(
        time=np.array([0.0, 0.1]),
        marker_names=("C7", "L.Shoulder", "R.Shoulder"),
        marker_xyz=np.zeros((2, 3, 3)),
        impact_idx=1,
        events=(),
        source=SourceProvenance("dummy.c3d", "c3d", "TW", "swing", "0000")
    )
    
    target = MultiSourceTarget(club=club, body=body)
    
    # Mock _body_target_to_json_file to intercept the file creation and check it
    original_to_json = None
    import simscape_adapter
    original_to_json = simscape_adapter._body_target_to_json_file
    
    captured_json_data = {}
    captured_file_path = None
    
    def mock_to_json(b_target):
        nonlocal captured_json_data, captured_file_path
        path = original_to_json(b_target)
        captured_file_path = path
        with open(path, "r") as f:
            captured_json_data = json.load(f)
        return path
        
    import sys
    with patch.dict(sys.modules, {"matlab": MagicMock()}):
        with patch("simscape_adapter._body_target_to_json_file", side_effect=mock_to_json):
            cost = adapter.compute_cost(np.zeros(7), target)
    
    assert cost == 42.0
    
    # Verify the JSON was parsed and written correctly
    assert captured_file_path is not None
    assert "time_s" in captured_json_data
    assert "marker_names" in captured_json_data
    assert "marker_xyz" in captured_json_data
    
    # The time_s should have 2 elements
    assert len(captured_json_data["time_s"]) == 2
    # The marker_xyz should be 2 frames of 2 markers of 3 coords
    assert len(captured_json_data["marker_xyz"]) == 2
    assert len(captured_json_data["marker_xyz"][0]) == 3
    assert len(captured_json_data["marker_xyz"][0][0]) == 3
    
    # Verify load_body_target_json was called on the matlab engine
    mock_eng.load_body_target_json.assert_called_once_with(captured_file_path, nargout=1)
    
    import os
    # Verify the file was cleaned up by the finally block
    assert not os.path.exists(captured_file_path)
