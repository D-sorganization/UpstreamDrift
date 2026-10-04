"""Real scoped capture lineage for bounded read-group tests; no native SDK."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from hypothesis_fixture import imported_capture
from test_scope_fixtures import fixture_review_artifact
from src.shared.python.workspace.necromatcher_capture_identity import capture_identity
from src.shared.python.workspace import necromatcher_fit as fit
from src.shared.python.workspace.necromatcher_fit_jobs import NativeRefitOptions


def make_scoped_refit_case(
    fit_case: Any, tmp_path: Path
) -> tuple[Any, NativeRefitOptions, dict[str, Any]]:
    library, source, original = fit_case
    asset = imported_capture(library, tmp_path / "source-frames")
    identity = capture_identity(library, asset.dataset_id)
    reference = fixture_review_artifact(tmp_path, identity, 0, 3)
    library.add_source_scope_review("review", "practice", Path(reference.path))
    scope = library.load_source_scope_review("review")
    payload = deepcopy(original)
    payload.update(
        capture_id=asset.dataset_id,
        capture_hash=asset.metadata["hash"],
        frame_indices=[0, 1, 2],
        frames=[f.to_dict() for f in identity.frames],
        q=[[0.1], [0.15], [0.2]],
    )
    source.write_text(json.dumps(payload), encoding="utf-8")
    library.add_fit("grandparent", "practice", source)
    options = NativeRefitOptions((0, 2), 2, (1.0,))
    bound = fit.admit_refit_scope(
        library, payload, (0, 2), options.config, requested=scope
    )
    payload["provenance"].update(
        source_fit_scope=scope.to_record(),
        source_fit_scope_binding=fit.scope_binding_record(bound, (0, 2)),
        request_options=json.loads(json.dumps(asdict(options))),
    )
    payload["evidence"]["original_fit"] = {
        "config": asdict(options.config),
        "frame_indices": [0, 2],
        "source_times": [0, 0.2],
    }
    for parent_id, new_id in (("grandparent", "parent"), ("parent", "child")):
        payload["provenance"].update(
            warm_start_fit_id=parent_id,
            warm_start_fit_hash=library.load_asset(parent_id).metadata["hash"],
        )
        source.write_text(json.dumps(payload), encoding="utf-8")
        library.add_fit(new_id, "practice", source)
    return library, options, library.load_fit("child")
