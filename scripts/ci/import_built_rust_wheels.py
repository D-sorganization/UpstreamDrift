"""Import every Rust wheel built by the CI Maturin lane.

The script fails closed: each requested module must import and expose at least
one expected binding. CI installs the freshly built wheels before running this
so imports come from wheel artifacts, not the source tree.
"""

from __future__ import annotations

import argparse
import importlib
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType


EXPECTED_BINDINGS: dict[str, tuple[str, ...]] = {
    "upstream_physics": (
        "IntegratorConfig",
        "ContactParameters",
        "simulate_ball_trajectory_py",
    ),
    "upstream_mocap_preproc": ("butterworth_filter", "savgol_filter", "resample_fps"),
    "upstream_mocap_io": ("parse_c3d", "parse_trc", "parse_bvh"),
    "upstream_muscle": ("f_l", "f_v", "HillMuscleModel"),
    "upstream_motion_matching": ("finite_diff_q_to_qdot_qddot",),
    "ai_backend": ("AIConfig", "AIEngine", "RagPipeline"),
}


def smoke_module(module_name: str, module: ModuleType) -> None:
    """Run a minimal backend call through the imported extension module."""
    if module_name == "upstream_physics":
        module.IntegratorConfig(dt=0.01, max_steps=10)
        module.ContactParameters(cor=0.8, friction=0.3)
    elif module_name == "upstream_mocap_preproc":
        import numpy as np

        data = np.arange(12.0, dtype=float).reshape(2, 2, 3)
        source_timestamps = np.array([0.0, 1.0], dtype=float)
        target_timestamps = np.array([0.0, 0.5, 1.0], dtype=float)
        result = module.resample_fps(data, source_timestamps, target_timestamps)
        if result.shape != (3, 2, 3):
            raise RuntimeError(f"upstream_mocap_preproc smoke returned {result.shape}")
    elif module_name == "upstream_mocap_io":
        missing = Path(tempfile.gettempdir()) / "upstreamdrift_missing_smoke.trc"
        try:
            module.parse_trc(missing)
        except Exception as exc:  # noqa: BLE001 - controlled backend error smoke.
            if "TRC parse error" not in str(exc):
                raise RuntimeError(
                    f"upstream_mocap_io smoke failed unexpectedly: {exc}"
                ) from exc
        else:
            raise RuntimeError(
                "upstream_mocap_io parse_trc unexpectedly accepted missing file"
            )
    elif module_name == "upstream_muscle":
        if not 0.0 < float(module.f_l(1.0)) <= 1.0:
            raise RuntimeError("upstream_muscle f_l smoke returned an invalid value")
    elif module_name == "upstream_motion_matching":
        qdot, qddot = module.finite_diff_q_to_qdot_qddot(
            [[0.0], [1.0], [2.0]],
            0.5,
        )
        if len(qdot) != 3 or len(qddot) != 3:
            raise RuntimeError(
                "upstream_motion_matching finite-diff smoke returned wrong shape"
            )
    elif module_name == "ai_backend":
        cfg = module.AIConfig("k", "https://api.example/v1", "m", ":memory:")
        if cfg.chat_url() != "https://api.example/v1/chat/completions":
            raise RuntimeError("ai_backend AIConfig smoke returned wrong chat URL")


def verify_module(module_name: str) -> None:
    """Import a built extension module and assert its public smoke surface."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - this is a fail-closed CI probe.
        raise RuntimeError(
            f"failed to import Rust wheel module {module_name!r}: {exc}"
        ) from exc

    missing = [
        name for name in EXPECTED_BINDINGS[module_name] if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            f"Rust wheel module {module_name!r} imported but is missing bindings: "
            f"{', '.join(missing)}"
        )
    smoke_module(module_name, module)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "modules",
        nargs="+",
        choices=sorted(EXPECTED_BINDINGS),
        help="Python extension module names installed from built Rust wheels.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for module_name in args.modules:
        verify_module(module_name)
        print(f"verified Rust wheel import: {module_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
