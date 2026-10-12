"""Independent screw-motion research replay, no engine or disk writes."""

import json
from contextlib import contextmanager
from typing import Any

import numpy as np
from src.shared.python.motion_matching.pipeline import MarkerLinearization
from src.shared.python.simulation_backends import Trace
from src.shared.python.workspace.necromatcher_native import NativeFitBinding
from src.shared.python.workspace.project_store import DatasetMetadata


class ScrewPlant:
    coordinate_order = ("translation", "rotation")
    coordinate_units = ("m", "rad")
    engine_name = "synthetic"

    def create_marker_linearizer(self, attachments: dict) -> Any:
        plant = self

        class Linearizer:
            def marker_linearization(self, q: np.ndarray) -> MarkerLinearization:
                theta = q[1]
                c, s = np.cos(theta), np.sin(theta)
                r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                positions, jacobian = [], []
                for body, point in attachments.values():
                    assert body == "club"
                    arm = r @ np.array(point)
                    positions.append(arm + [q[0], 0, 0])
                    jacobian.append(
                        np.column_stack(([1, 0, 0], np.cross([0, 0, 1], arm)))
                    )
                return MarkerLinearization(
                    np.array(positions),
                    np.array(jacobian),
                    tuple(attachments),
                    plant.coordinate_order,
                )

        return Linearizer()

    def frame_poses(self, mapping: dict, q: np.ndarray) -> dict:
        c, s = np.cos(q[1]), np.sin(q[1])
        return {
            "club": (
                np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
                np.array([q[0], 0, 0]),
            )
        }


class ReplayLibrary:
    def __init__(self) -> None:
        self.reads = []
        self.active = False
        kinds = {
            "replay": "authored_replay",
            "profile": "torque_profile",
            "fit": "kinematic_fit",
            "model": "native_model",
            "capture": "image_capture",
        }
        self.assets = {
            name: DatasetMetadata(
                name, "swing", "unused", kind, {"hash": "sha256:" + str(i) * 64}
            )
            for i, (name, kind) in enumerate(kinds.items())
        }
        meta = {
            "schema": "necromatcher/authored-replay/1",
            "scientific_qualified": False,
            "physical_source_time_qualified": False,
            "independent_replay_executed": True,
            "source_frame_index": 17,
            "source_frame_json": json.dumps({"frame_id": "source-17"}),
            "coordinate_order_json": json.dumps(["translation", "rotation"]),
            "coordinate_units_json": json.dumps(["m", "rad"]),
            "effort_units_json": json.dumps(["N", "N*m"]),
        }
        for name in ("profile", "fit", "model", "capture"):
            meta[name + "_id"] = name
            meta[name + "_hash"] = self.assets[name].metadata["hash"]
        self.trace = Trace(
            np.array([2.0, 2.1]),
            np.array([[0, 0], [0.2, np.pi / 2]]),
            np.array([[2.0, 3.0], [2.0, 3.0]]),
            dt=0.1,
            backend="synthetic",
            meta=meta,
        )
        self.fit = {
            "coordinate_order": ["translation", "rotation"],
            "coordinate_units": ["m", "rad"],
            "model_id": "model",
            "model_hash": meta["model_hash"],
            "capture_id": "capture",
            "capture_hash": meta["capture_hash"],
            "provenance": {"native_definition": {"bodies": [{"name": "club"}]}},
        }
        self.binding = NativeFitBinding(
            "fit",
            meta["fit_hash"],
            "model",
            meta["model_hash"],
            self.fit,
            ScrewPlant(),
            ("m", "rad"),
        )

    @contextmanager
    def authenticated_read(self):
        assert not self.active
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def load_asset(self, name: str) -> DatasetMetadata:
        assert self.active
        self.reads.append(name)
        return self.assets[name]

    def load_replay(self, name: str) -> Trace:
        assert self.active and name == "replay"
        return self.trace

    def load_fit(self, name: str) -> dict:
        assert self.active and name == "fit"
        return self.fit


def impact_case(monkeypatch) -> ReplayLibrary:
    from src.shared.python.workspace import necromatcher_impact

    library = ReplayLibrary()
    monkeypatch.setattr(
        necromatcher_impact, "load_native_fit_binding", lambda *args: library.binding
    )
    return library
