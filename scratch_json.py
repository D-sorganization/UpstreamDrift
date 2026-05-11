import matlab.engine
eng = matlab.engine.start_matlab()
import json
payload = {
    "schema": "body_target_json_v1",
    "time_s": [0.0, 0.1],
    "marker_names": ["pelvis"],
    "marker_xyz": [[[0.0, 0.0, 0.0]], [[0.1, 0.1, 0.1]]],
    "impact_idx": 0,
    "events": [],
    "source": {"filename": "x"},
    "coordinate_frame": "z_up_right_handed",
}
j = json.dumps(payload)
out = eng.jsondecode(j)
print(type(out))
print(out)
eng.quit()
