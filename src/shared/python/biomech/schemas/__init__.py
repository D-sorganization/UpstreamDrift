"""JSON schemas for the biomech shared-folder convention.

The canonical schema is :data:`MODEL_PACK_V1_SCHEMA_PATH`, used by sibling
biomech repos to validate their `model_pack.yaml` / `tool_pack.yaml`
manifests at build time and by UpstreamDrift's discovery layer at runtime.
"""

from __future__ import annotations

from pathlib import Path

MODEL_PACK_V1_SCHEMA_PATH: Path = Path(__file__).resolve().parent / "model_pack_v1.json"
"""Absolute filesystem path to the bundled draft 2020-12 JSON Schema."""


def load_model_pack_v1_schema() -> dict:
    """Return the bundled `model_pack/v1` schema as a Python dict."""
    import json

    return json.loads(MODEL_PACK_V1_SCHEMA_PATH.read_text(encoding="utf-8"))


__all__ = ["MODEL_PACK_V1_SCHEMA_PATH", "load_model_pack_v1_schema"]
