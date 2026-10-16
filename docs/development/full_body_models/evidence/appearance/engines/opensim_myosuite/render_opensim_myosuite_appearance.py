"""R9b (#12689) evidence: OpenSim and MyoSuite native stills with the shared appearance.

Renders a capture-A driver same-input bundle (spec + reference ``q``, from the
private swing datastore) at address and impact, face-on and down-the-line,
through the real native-viewer backends: OpenSim's simbody visualizer under
xvfb and MyoSuite's ``MJRenderer``. Both now draw the DEFAULT shared
appearance document (R8, #12685). Frames are clean: no HUD, no force glyphs.

Usage, from the repository root (Linux; OpenSim needs xvfb-run, xwd and
xwininfo; MyoSuite may run in its own interpreter)::

    PYTHONPATH=.:src NATIVE_VIEWER_MYOSUITE_PYTHON=<myosuite python> python3 \\
docs/development/full_body_models/evidence/appearance/engines/opensim_myosuite/\\
render_opensim_myosuite_appearance.py --bundle BUNDLE.npz --out OUT_DIR

Stills are visual evidence, not a pixel unit test; appearance never feeds
dynamics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ENGINES = ("opensim", "myosuite")
VIEWS = ("face_on", "down_the_line")
WIDTH, HEIGHT = 960, 720


def phase_indices(bundle: object) -> dict[str, int]:
    """Address (first sample) and impact (bundle impact split) sample indices."""
    provenance = getattr(bundle, "provenance", {}) or {}
    impact_s = provenance.get("impact_split_time_s")
    if impact_s is None:
        raise ValueError("bundle provenance has no impact_split_time_s")
    last = int(np.asarray(bundle.reference_q).shape[0]) - 1  # type: ignore[attr-defined]
    impact = min(last, int(round(float(impact_s) / float(bundle.dt_s))))  # type: ignore[attr-defined]
    return {"address": 0, "impact": impact}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engines", default=",".join(ENGINES))
    args = parser.parse_args()
    import imageio.v2 as imageio

    from src.tools.native_viewer_export.backends.registry import make_backend
    from src.tools.native_viewer_export.core import ExportSettings, load_swing_input

    swing = load_swing_input(args.bundle, swing="driver", club="Driver")
    phases = phase_indices(swing.bundle)
    settings = ExportSettings(
        views=VIEWS,
        width=WIDTH,
        height=HEIGHT,
        hud=False,
        overlays=False,
        ball=False,
        multiview=False,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for engine in args.engines.split(","):
        backend = make_backend(engine)
        reason = backend.unavailable_reason()
        if reason is not None:
            raise RuntimeError(f"{engine} backend unavailable: {reason}")
        frames = backend.render(swing, settings, list(phases.values()), None)
        for phase, views in zip(phases, frames, strict=True):
            for view, image in views.items():
                path = args.out / f"r9b_driver_{engine}_{view}_{phase}.png"
                imageio.imwrite(path, np.asarray(image))
                print(path)  # noqa: T201 - evidence script progress


if __name__ == "__main__":
    main()
