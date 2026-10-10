"""Emit a bounded native diagnostic receipt without copying source models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
    build_musculoskeletal_model,
)
from src.engines.physics_engines.opensim.python.native_mtp_reduction import (
    ZeroMtpReductionRequest,
    derive_zero_mtp_model,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reduce(source: Path, output: Path) -> dict[str, object]:
    receipt = derive_zero_mtp_model(
        ZeroMtpReductionRequest(
            source_model_path=source,
            source_sha256=_sha(source),
            derived_model_path=output,
            declared_target_rad=(("mtp_angle_r", 0.0), ("mtp_angle_l", 0.0)),
        )
    )
    return asdict(receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--golf-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise ValueError("receipt parent directory missing")
    with TemporaryDirectory(prefix="native-mtp-reduction-") as temporary:
        scratch = Path(temporary)
        donor = _reduce(args.donor, scratch / "donor_derived.osim")
        factory, _ = build_musculoskeletal_model(
            args.golf_model, base_model_path=args.donor
        )
        factory_source = scratch / "factory_source.osim"
        factory.printToXML(str(factory_source))
        mixed = _reduce(factory_source, scratch / "factory_derived.osim")
    payload = {
        "issue": 12150,
        "scope": "diagnostic-zero-mtp-reduction-not-physiology-or-replay-qualification",
        "donor": donor,
        "scaled_club_factory": mixed,
        "golf_model_sha256": _sha(args.golf_model),
        "receipt_script_sha256": _sha(Path(__file__)),
        "external_resource_closure": "unverified-visual-mesh-warnings-retained",
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
