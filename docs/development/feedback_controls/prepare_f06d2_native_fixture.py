"""Package a trusted R2025b diagnostic producer through the owned-byte API.

This developer harness neither accepts arbitrary MAT producers nor qualifies a
production model. The native reference tests execute the resulting archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.engines.native_replay_contracts import native_replay_contract_types
from src.engines.Simscape_Multibody_Models.python import (
    SimscapeRestartBinding,
    owned_simscape_replay_files,
)

PROVIDER_FILES = (
    "capture_native_simscape_execution.m",
    "load_native_simscape_snapshot.m",
    "native_simscape_bytes_sha256.m",
    "native_simscape_file_sha256.m",
    "native_simscape_provider_sha256.m",
    "run_native_simscape_restart.m",
    "run_owned_native_simscape_replay.m",
)


def _verify_producer(directory: Path) -> tuple[dict[str, Any], bytes, bytes]:
    receipt = json.loads(
        (directory / "native-restart-receipt.json").read_text(encoding="utf-8")
    )
    source = (directory / "native_restart_fixture_11921.slx").read_bytes()
    payload = (directory / "native-operating-point.mat").read_bytes()
    for content, expected in (
        (source, receipt["model_sha256"]),
        (payload, receipt["snapshot_sha256"]),
    ):
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("trusted native producer artifact digest differs")
    records = "".join(
        f"{name}:{hashlib.sha256((ROOT / 'scripts/matlab' / name).read_bytes()).hexdigest()}\n"
        for name in sorted(PROVIDER_FILES)
    ).encode("utf-8")
    if hashlib.sha256(records).hexdigest() != receipt["provider_sha256"]:
        raise ValueError("native producer source differs from this checkout")
    return receipt, source, payload


def _history(directory: Path, changed: bool, contracts: Any) -> Any:
    with (directory / "saved-input.csv").open(newline="", encoding="utf-8") as stream:
        original = [
            (float(row["time_s"]), float(row["value"]))
            for row in csv.DictReader(stream)
        ]
    tail = [(time, force) for time, force in original if time >= 0.2]
    if len(tail) != 21:
        raise ValueError("expected the trusted diagnostic's 21-sample suffix")
    values = tuple(
        (((0.8 if time < 0.3 else -0.2) if changed else force),) for time, force in tail
    )
    return contracts.InputHistory(
        contracts.ActuationInputKind.ACTUATOR_FORCE,
        "simulation_relative",
        contracts.InputInterpolation.ZERO_ORDER_HOLD,
        tuple(index / 100 for index in range(len(tail))),
        (
            contracts.InputChannel(
                "force",
                "native_force_input",
                "N",
                frame_id="fixture-translational-axis",
            ),
        ),
        values,
    )


def _declaration(
    receipt: dict[str, Any], payload: bytes, history: Any, contracts: Any
) -> tuple[Any, SimscapeRestartBinding]:
    native = receipt["execution_binding"]
    identity = contracts.NativeStateIdentity(
        "simscape",
        native["runtime_id"],
        receipt["provider_sha256"],
        receipt["model_sha256"],
        receipt["model_sha256"],
    )
    execution = contracts.NativeStateExecution(
        "ode23t",
        native["solver_version"],
        "simscape-model-operating-point",
        "1.0.0",
        native["effective_configuration_sha256"],
        native["compatibility_sha256"],
    )
    artifact = contracts.NativeStateArtifact(
        contracts.NativeStateEncoding(
            "matlab-mat-v7.3", "7.3.0", "Simulink.op.ModelOperatingPoint"
        ),
        identity,
        contracts.NativeStateClock(0.0, 0.2, "simulation_absolute"),
        execution,
        receipt["snapshot_sha256"],
        len(payload),
        contracts.NativeStateRole.COMPLETE_NATIVE_RESTART,
    )
    rows = "".join(
        f"{0.2 + time:.17g},{value[0]:.17g}\n"
        for time, value in zip(history.time_seconds, history.values, strict=True)
    ).encode("ascii")
    policy = contracts.ReplayExecutionPolicy(
        contracts.ReplayMode.EXTERNALLY_FORCED,
        "ode23t",
        native["solver_version"],
        "ode23t",
        "adaptive",
        None,
        "simscape-model-operating-point",
        "1.0.0",
        "simscape-from-workspace-zoh",
        "1.0.0",
        False,
        False,
        False,
        external_loads_sha256=hashlib.sha256(rows).hexdigest(),
    )
    envelope = contracts.build_native_state_replay_envelope(
        "native-simscape-owned-diagnostic-11942",
        contracts.NativeReplayModel(
            "simscape/native_restart_fixture_11921", "diagnostic", "1.0.0", ("force",)
        ),
        artifact,
        (
            contracts.CapabilityDeclaration(
                "native_restart",
                True,
                contracts.CapabilitySupport.SUPPORTED,
                contracts.CapabilityAvailability.AVAILABLE,
            ),
        ),
        history,
        policy,
    )
    return envelope, SimscapeRestartBinding(
        "native_restart_fixture_11921",
        identity,
        execution,
        receipt["saved_input_sha256"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_archive", type=Path)
    parser.add_argument("--changed-future", action="store_true")
    args = parser.parse_args()
    receipt, source, payload = _verify_producer(args.source_directory)
    contracts = native_replay_contract_types()
    history = _history(args.source_directory, args.changed_future, contracts)
    envelope, binding = _declaration(receipt, payload, history, contracts)
    owned = contracts.freeze_native_state_artifact(envelope.artifact, payload)
    with owned_simscape_replay_files(envelope, owned, source, binding) as directory:
        request_sha256 = hashlib.sha256(
            (directory / "native-request.json").read_bytes()
        ).hexdigest()
        with zipfile.ZipFile(args.output_archive, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(directory.iterdir()):
                archive.write(path, path.name)
        # This digest must be passed out of band to the native consumer.
        sys.stdout.write(request_sha256 + "\n")


if __name__ == "__main__":
    main()
