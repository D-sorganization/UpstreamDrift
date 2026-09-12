from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class _PlaybackMixin:
    data: dict[str, Any]
    current_idx: int
    engine: Any
    run_id: str

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]:
        if field_name is None:
            raise ValueError("field_name must be provided")
        if field_name not in self.data:
            return np.array([]), np.array([])

        values: Any = self.data[field_name]

        if values is None or self.current_idx == 0:
            return np.array([]), np.array([])

        times = self.data["times"][: self.current_idx]

        if isinstance(values, np.ndarray):
            return times, values[: self.current_idx]
        if isinstance(values, list):
            return times, np.array(values[: self.current_idx])
        return times, values

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        if source_name is None:
            raise ValueError("source_name must be provided")
        if source_name not in self.data["induced_accelerations"]:
            logger.warning(
                "Induced acceleration source '%s' not found in recorded data. "
                "Available sources: %s. Returning empty series.",
                source_name,
                list(self.data["induced_accelerations"].keys()),
            )
            return np.array([]), np.array([])

        data = self.data["induced_accelerations"][source_name]

        if isinstance(data, np.ndarray):
            return self.data["times"][: self.current_idx], data[: self.current_idx]

        result: tuple[np.ndarray, np.ndarray] = data
        return result

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        if cf_name is None:
            raise ValueError("cf_name must be provided")
        if cf_name not in self.data["counterfactuals"]:
            return np.array([]), np.array([])
        result: tuple[np.ndarray, np.ndarray] = self.data["counterfactuals"][cf_name]
        return result

    def _resolve_engine_name(self) -> str:
        """Resolve a human-readable engine name from the engine instance."""
        if hasattr(self.engine, "get_capabilities"):
            try:
                caps = self.engine.get_capabilities()
                if hasattr(caps, "engine_name") and caps.engine_name:
                    return str(caps.engine_name)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass
        return type(self.engine).__name__

    def get_data_dict(self) -> dict[str, Any]:
        export_data: dict[str, Any] = {}
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                export_data[k] = v[: self.current_idx] if v.ndim > 0 else v
            elif isinstance(v, list) and v:
                try:
                    export_data[k] = np.array(v)
                except (ValueError, TypeError, RuntimeError) as e:
                    logger.debug("Failed to convert list '%s' to numpy array: %s", k, e)
                    export_data[k] = v
            else:
                export_data[k] = v

        export_data["model_name"] = getattr(self.engine, "model_name", "")
        export_data["num_frames"] = self.current_idx

        # Provenance metadata (#8820)
        engine_name = self._resolve_engine_name()
        run_id = getattr(self, "run_id", None)
        model_path = getattr(self.engine, "model_path", None)

        export_data["engine_name"] = engine_name
        if run_id:
            export_data["run_id"] = run_id
        if model_path:
            export_data["model_file_path"] = str(model_path)

        from src.shared.python.data_io.provenance import ProvenanceInfo

        provenance = ProvenanceInfo.capture(
            model_path=model_path,
            engine_name=engine_name,
            run_id=run_id,
            parameters={"num_frames": self.current_idx},
        )
        export_data["provenance"] = provenance
        if provenance.model_file_hash:
            export_data["model_file_hash"] = provenance.model_file_hash

        return export_data
