"""Map each release figure to the stored run that wrote it (Issue #14518).

Every figure under a release's ``figures`` directory gets one entry: the
preset and run id of the first listed run that has it as an output, the
SHA-256 of that run's copy, whether it equals the committed release file, and
when it does not, why. ``commit_bound`` marks a copy that embeds its run's
short commit SHA and so reproduces only at that commit. The reason is read from the matplotlib version both
files embed, so an entry that does not match explains itself; a figure no
listed run wrote is recorded as unmapped rather than left out.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import json
from pathlib import Path
import re
from typing import Any

from src.sim_run.manifest import RunManifest
from src.sim_run.manifest_emitter import sha256_file
from src.sim_run.run_registry import RunRegistry, default_store_root

SCHEMA_VERSION = 1
UNMAPPED = "no stored run writes this figure"
# git's default --short length; a copy holding it reproduces only at that commit.
SHORT_SHA = 7
_MATPLOTLIB = re.compile(rb"Matplotlib(?: pdf backend)? v(\d+\.\d+\.\d+)")


def _matplotlib_version(path: Path) -> str | None:
    match = _MATPLOTLIB.search(path.read_bytes())
    return match.group(1).decode("ascii") if match else None


def _mismatch_reason(run_copy: Path, release: Path) -> str:
    ran, released = _matplotlib_version(run_copy), _matplotlib_version(release)
    if ran is None and released is None:
        return "bytes differ; neither copy records its matplotlib version"
    if ran != released:
        return f"matplotlib {ran} at run vs {released} at release"
    return (
        f"bytes differ at the same matplotlib {ran}: "
        "input data or plotting code changed since the release"
    )


def _entry(
    registry: RunRegistry, run_id: str, manifest: RunManifest, name: str, release: Path
) -> dict[str, Any]:
    run_sha256 = manifest.outputs[name]
    run_copy = registry.output_path(run_id, name)
    matches = run_sha256 == sha256_file(release)
    reason = None if matches else _mismatch_reason(run_copy, release)
    short_sha = manifest.ud_commit[:SHORT_SHA].encode("ascii")
    return {
        "commit_bound": short_sha in run_copy.read_bytes(),
        "preset": manifest.parameters.get("preset"),
        "run_id": run_id,
        "run_sha256": run_sha256,
        "matches_release": matches,
        "mismatch_reason": reason,
    }


def build_figure_run_map(
    registry: RunRegistry, run_ids: Iterable[str], *, repo_root: Path, figures: str
) -> dict[str, Any]:
    """Build the map for every file under ``repo_root / figures``.

    Raises:
        ValueError: if a run id is malformed.
        RunRegistryError: if a run id is not stored.
    """
    manifests = {run_id: registry.show(run_id) for run_id in run_ids}
    entries: dict[str, Any] = {}
    for path in sorted((Path(repo_root) / figures).rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(repo_root).as_posix()
        writer = next((r for r, m in manifests.items() if name in m.outputs), None)
        if writer is None:
            entries[name] = {
                "commit_bound": None,
                "preset": None,
                "run_id": None,
                "run_sha256": None,
                "matches_release": False,
                "mismatch_reason": UNMAPPED,
            }
        else:
            entries[name] = _entry(registry, writer, manifests[writer], name, path)
    cited = {entry["run_id"] for entry in entries.values()} - {None}
    return {
        "schema_version": SCHEMA_VERSION,
        "figures_root": figures,
        "runs": {
            run_id: {
                "preset": manifest.parameters.get("preset"),
                "ud_commit": manifest.ud_commit,
            }
            for run_id, manifest in sorted(manifests.items())
            if run_id in cited
        },
        "figures": entries,
    }


def figure_run_map_text(figure_map: dict[str, Any]) -> str:
    """Canonical JSON text: sorted keys, one line per run and per figure."""

    def block(key: str) -> str:
        rows = [
            f"    {json.dumps(name)}: {json.dumps(value, sort_keys=True)}"
            for name, value in sorted(figure_map[key].items())
        ]
        return f'  "{key}": {{\n' + ",\n".join(rows) + "\n  }"

    head = [
        f"  {json.dumps(key)}: {json.dumps(value)}"
        for key, value in sorted(figure_map.items())
        if key not in ("figures", "runs")
    ]
    return "{\n" + ",\n".join([*head, block("figures"), block("runs")]) + "\n}\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the map for the run ids listed one per line in ``--run-ids``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=None, help="run store root")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--figures", required=True, help="repo-relative figures dir")
    parser.add_argument("--run-ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_ids = args.run_ids.read_text(encoding="utf-8").split()
    figure_map = build_figure_run_map(
        RunRegistry(args.store or default_store_root()),
        run_ids,
        repo_root=args.repo_root,
        figures=args.figures,
    )
    args.out.write_text(figure_run_map_text(figure_map), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
