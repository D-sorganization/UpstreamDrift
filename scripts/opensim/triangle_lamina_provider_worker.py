"""Run the pinned Tools triangle-lamina calculation in an isolated process."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provider_identity(tools_src: Path) -> dict[str, str]:
    root = tools_src.resolve()
    if not (root / "shared" / "python").is_dir():
        raise ValueError("pinned Tools shared Python source is unavailable")
    sys.path.insert(0, str(root))
    triangle = importlib.import_module(
        "shared.python.humanoid_character_builder.mesh.triangle_lamina"
    )
    representation = importlib.import_module(
        "shared.python.humanoid_character_builder.mesh.inertia_calculator"
    )
    expected = {
        "provider": root
        / "shared/python/humanoid_character_builder/mesh/triangle_lamina.py",
        "representation": root
        / "shared/python/humanoid_character_builder/mesh/inertia_calculator.py",
    }
    triangle_file = triangle.__file__
    representation_file = representation.__file__
    if triangle_file is None or representation_file is None:
        raise ValueError("pinned Tools provider import has no source file")
    actual = {
        "provider": Path(triangle_file).resolve(),
        "representation": Path(representation_file).resolve(),
    }
    for label, path in actual.items():
        if path != expected[label].resolve():
            raise ValueError(f"pinned Tools {label} import resolved outside its source")
    return {
        "provider_sha256": _sha(actual["provider"]),
        "representation_sha256": _sha(actual["representation"]),
        "provider_path": str(actual["provider"]),
        "representation_path": str(actual["representation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-src", required=True, type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    identity = _provider_identity(args.tools_src)
    response: dict[str, object] = {
        **identity,
        "worker_sha256": _sha(Path(__file__).resolve()),
    }
    if args.input is not None:
        with np.load(args.input, allow_pickle=False) as payload:
            vertices = np.asarray(payload["vertices"], dtype=np.float64)
            faces = np.asarray(payload["faces"], dtype=np.int64)
            mass = float(payload["mass"])
        triangle = importlib.import_module(
            "shared.python.humanoid_character_builder.mesh.triangle_lamina"
        )
        result = triangle.compute_triangle_lamina(vertices, faces, mass)
        inertia = result.inertia
        response["result"] = {
            "center_m": list(inertia.center_of_mass),
            "inertia_kg_m2": inertia.as_matrix().tolist(),
            "area_m2": result.total_area_m2,
            "boundary_edges": result.boundary_edges,
            "nonmanifold_edges": result.nonmanifold_edges,
            "duplicate_faces": result.duplicate_positive_area_triangles,
        }
    print(json.dumps(response, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
