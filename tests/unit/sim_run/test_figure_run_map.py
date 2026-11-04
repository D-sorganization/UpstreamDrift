"""Release figure-to-run mapping (RUN-6, Issue #14518 / Epic #14436)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from src.sim_run.figure_run_map import (
    build_figure_run_map,
    figure_run_map_text,
)
from src.sim_run.manifest_emitter import (
    RunProvenance,
    emit_run_manifest,
    model_revision,
    sha256_file,
)
from src.sim_run.run_registry import RunRegistry, RunRegistryError

pytestmark = pytest.mark.unit

_FIGURES = "release/figures"
_OLD = "<svg><dc:title>Matplotlib v3.10.9, https://matplotlib.org/</dc:title>"


def _store_run(registry: RunRegistry, run_dir: Path, preset: str, files: dict) -> str:
    for name, text in files.items():
        (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_text(text)
    emit_run_manifest(
        run_dir,
        outputs=[run_dir / name for name in files],
        provenance=RunProvenance(
            model_id=f"release-preset/{preset}",
            revision=model_revision(preset),
            parameters={"preset": preset, "tier": "figure_script"},
            input_sha256="0" * 64,
            seed=0,
            engine_name="ud-preset",
            engine_version="python 3.12.13",
            ud_commit="a" * 40,
        ),
        created_at=datetime(2026, 11, 4, 16, 0, tzinfo=timezone.utc),
    )
    return registry.add(run_dir)


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, RunRegistry, list[str]]:
    root = tmp_path / "repo"
    figures = root / _FIGURES
    figures.mkdir(parents=True)
    (figures / "fig_same.svg").write_text(_OLD + "same</svg>")
    (figures / "fig_new.svg").write_text(_OLD + "old</svg>")
    (figures / "fig_drift.svg").write_text(_OLD + "before</svg>")
    (figures / "fig_diagram.svg").write_text("<svg>hand drawn</svg>")
    registry = RunRegistry(tmp_path / "store")
    first = _store_run(
        registry,
        tmp_path / "run1",
        "study_one",
        {
            f"{_FIGURES}/fig_same.svg": _OLD + "same</svg>",
            f"{_FIGURES}/fig_new.svg": _OLD.replace("3.10.9", "3.11.2") + "new</svg>",
        },
    )
    second = _store_run(
        registry,
        tmp_path / "run2",
        "study_two",
        {f"{_FIGURES}/fig_drift.svg": _OLD + "after</svg>"},
    )
    return root, registry, [first, second]


def _entries(setup) -> dict:
    root, registry, run_ids = setup
    return build_figure_run_map(registry, run_ids, repo_root=root, figures=_FIGURES)


def test_every_release_figure_has_an_entry(setup) -> None:
    figures = _entries(setup)["figures"]
    assert sorted(figures) == [
        f"{_FIGURES}/fig_diagram.svg",
        f"{_FIGURES}/fig_drift.svg",
        f"{_FIGURES}/fig_new.svg",
        f"{_FIGURES}/fig_same.svg",
    ]


def test_matching_figure_cites_its_run(setup) -> None:
    root, _, run_ids = setup
    entry = _entries(setup)["figures"][f"{_FIGURES}/fig_same.svg"]
    assert entry["preset"] == "study_one"
    assert entry["run_id"] == run_ids[0]
    assert entry["run_sha256"] == sha256_file(root / _FIGURES / "fig_same.svg")
    assert entry["matches_release"] is True
    assert entry["mismatch_reason"] is None
    assert entry["model_id"] == "release-preset/study_one"
    assert entry["tier"] == "figure_script"


def test_renderer_mismatch_explains_itself(setup) -> None:
    entry = _entries(setup)["figures"][f"{_FIGURES}/fig_new.svg"]
    assert entry["matches_release"] is False
    assert entry["mismatch_reason"] == "matplotlib 3.11.2 at run vs 3.10.9 at release"


def test_same_renderer_mismatch_explains_itself(setup) -> None:
    entry = _entries(setup)["figures"][f"{_FIGURES}/fig_drift.svg"]
    assert entry["run_id"] == setup[2][1]
    assert entry["matches_release"] is False
    assert entry["mismatch_reason"] == (
        "bytes differ at the same matplotlib 3.10.9; not attributed: data, "
        "code, or unpinned render metadata such as dates or SVG ids"
    )


def test_figure_without_a_run_is_unmapped(setup) -> None:
    entry = _entries(setup)["figures"][f"{_FIGURES}/fig_diagram.svg"]
    assert entry["run_id"] is None and entry["preset"] is None
    assert entry["model_id"] is None and entry["tier"] is None
    assert entry["matches_release"] is False
    assert entry["mismatch_reason"] == "no stored run writes this figure"


def test_runs_section_records_each_cited_run(setup) -> None:
    runs = _entries(setup)["runs"]
    first, second = setup[2]
    assert runs[first] == {"preset": "study_one", "ud_commit": "a" * 40}
    assert set(runs) == {first, second}


def test_first_listed_run_wins_for_a_shared_figure(setup, tmp_path) -> None:
    root, registry, run_ids = setup
    late = _store_run(
        registry,
        tmp_path / "run3",
        "study_three",
        {f"{_FIGURES}/fig_same.svg": _OLD + "other</svg>"},
    )
    entries = build_figure_run_map(
        registry, [*run_ids, late], repo_root=root, figures=_FIGURES
    )
    assert entries["figures"][f"{_FIGURES}/fig_same.svg"]["run_id"] == run_ids[0]


def test_unknown_run_id_is_rejected(setup) -> None:
    root, registry, _ = setup
    with pytest.raises(RunRegistryError, match="unknown run"):
        build_figure_run_map(registry, ["b" * 64], repo_root=root, figures=_FIGURES)


def test_text_is_canonical_json_with_one_line_per_figure(setup) -> None:
    entries = _entries(setup)
    text = figure_run_map_text(entries)
    assert json.loads(text) == entries
    lines = text.splitlines()
    assert sum(1 for line in lines if line.lstrip().startswith(f'"{_FIGURES}')) == 4
    assert text == figure_run_map_text(json.loads(text))


def test_figure_embedding_the_run_commit_is_commit_bound(setup, tmp_path) -> None:
    root, registry, run_ids = setup
    (root / _FIGURES / "fig_sha.svg").write_text(_OLD + "aaaaaaa</svg>")
    bound = _store_run(
        registry,
        tmp_path / "run4",
        "study_four",
        {f"{_FIGURES}/fig_sha.svg": _OLD + "git aaaaaaa</svg>"},
    )
    figures = build_figure_run_map(
        registry, [*run_ids, bound], repo_root=root, figures=_FIGURES
    )["figures"]
    assert figures[f"{_FIGURES}/fig_sha.svg"]["commit_bound"] is True
    assert figures[f"{_FIGURES}/fig_same.svg"]["commit_bound"] is False
    assert figures[f"{_FIGURES}/fig_diagram.svg"]["commit_bound"] is None


def test_main_writes_the_map_from_a_run_id_file(setup, tmp_path) -> None:
    from src.sim_run.figure_run_map import main

    root, registry, run_ids = setup
    ids = tmp_path / "run_ids.txt"
    ids.write_text("\n".join(run_ids) + "\n")
    out = root / "release" / "figure_run_map.json"
    argv = ["--store", str(tmp_path / "store"), "--repo-root", str(root)]
    argv += ["--figures", _FIGURES, "--run-ids", str(ids), "--out", str(out)]
    assert main(argv) == 0
    assert json.loads(out.read_text()) == _entries(setup)


def test_mismatch_without_recorded_renderer_says_so(setup, tmp_path) -> None:
    root, registry, run_ids = setup
    (root / _FIGURES / "fig_plain.svg").write_text("<svg>before</svg>")
    plain = _store_run(
        registry,
        tmp_path / "run5",
        "study_five",
        {f"{_FIGURES}/fig_plain.svg": "<svg>after</svg>"},
    )
    figures = build_figure_run_map(
        registry, [*run_ids, plain], repo_root=root, figures=_FIGURES
    )["figures"]
    assert figures[f"{_FIGURES}/fig_plain.svg"]["mismatch_reason"] == (
        "bytes differ; neither copy records its matplotlib version"
    )
