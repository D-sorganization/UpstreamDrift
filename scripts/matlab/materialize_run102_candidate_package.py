"""Materialize MS-60 run-102 candidate.npz, playback.gif, and run_manifest.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.shared.python.motion_matching.candidate_convert import (
    convert_simscape_returned_replay,
)
from src.shared.python.motion_matching.candidate_io import save_candidate
from src.shared.python.motion_matching.export import export_video
from src.shared.python.motion_matching.simscape_run_manifest import (
    SimscapeRunManifestRequest,
    build_simscape_run_manifest,
)


def main() -> None:
    evidence = Path(
        "docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_102"
    )
    candidate_doc = json.loads(
        (evidence / "returned-candidate.json").read_text(encoding="utf-8")
    )
    qualified = json.loads(
        (evidence / "qualified_candidate_replay.json").read_text(encoding="utf-8")
    )
    receipt = json.loads((evidence / "receipt.json").read_text(encoding="utf-8"))

    replay_path = evidence / "returned-replay.npz"
    replay_sha = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    candidate_sha = str(
        receipt.get("returned_sha256")
        or hashlib.sha256(
            (evidence / "returned-candidate.json").read_bytes()
        ).hexdigest()
    )

    cand = convert_simscape_returned_replay(
        replay_path, candidate_doc=candidate_doc, engine="simscape"
    )
    cand_path = evidence / "candidate.npz"
    save_candidate(cand, cand_path)
    print(f"wrote {cand_path} size={cand_path.stat().st_size}")

    gif_path = evidence / "playback.gif"
    export_video(cand, engine="simscape", path=gif_path, stride=4)
    print(f"wrote {gif_path} size={gif_path.stat().st_size}")

    manifest = build_simscape_run_manifest(
        SimscapeRunManifestRequest(
            run_id="two_window_fit_9967_102",
            matlab_release=qualified["matlab_release"],
            matlab_version=qualified["matlab_version"],
            host="DeskComputer",
            machine="DeskComputer",
            model_sha256=candidate_doc["model_sha256"],
            candidate_sha256=candidate_sha,
            replay_npz_sha256=replay_sha,
            wall_clock_s=float(qualified["elapsed_s"]),
            qualification=qualified["qualification"],
            evidence_dir=(
                "docs/development/simscape_tour_matching/native_evidence/"
                "two_window_fit_9967_102"
            ),
            artifacts={
                "candidate_npz": "candidate.npz",
                "playback_gif": "playback.gif",
                "returned_replay_npz": "returned-replay.npz",
                "returned_candidate_json": "returned-candidate.json",
                "qualified_replay_json": "qualified_candidate_replay.json",
                "qualified_replay_mat": "qualified_candidate_replay.mat",
                "replay_script": "replay_returned102_r2025b.m",
            },
            issue="#10347",
            extra={
                "metrics_mm": {
                    "whole_rms_mm": qualified["metrics"]["whole_rms_mm"],
                    "early_rms_mm": qualified["metrics"]["early_rms_mm"],
                    "terminal_rms_mm": qualified["metrics"]["terminal_rms_mm"],
                    "club_cluster_rms_mm": qualified["metrics"]["club_cluster_rms_mm"],
                    "pelvis_yaw_error_pct": qualified["metrics"][
                        "pelvis_yaw_error_pct"
                    ],
                },
                "cross_engine_parity_mm": {
                    "max_euclidean_mm": qualified["cross_engine_parity"][
                        "max_marker_euclidean_discrepancy_mm"
                    ]
                },
                "gates": qualified["gates"],
                "solver_configuration": qualified["solver_configuration"],
                "note": (
                    "Native R2025b qualification was recorded on DeskComputer; "
                    "candidate.npz/playback.gif derived from committed "
                    "returned-replay.npz for CI-reproducible MatchedSwingCandidate "
                    "packaging (MS-60)."
                ),
            },
        )
    )
    manifest_path = evidence / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    print(f"candidate_sha={candidate_sha}")
    print(f"replay_sha={replay_sha}")


if __name__ == "__main__":
    main()
