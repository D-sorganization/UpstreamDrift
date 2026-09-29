"""Documentation freshness and engine capability-evidence authority (#9193)."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from scripts import companion_catalog, companion_evidence

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "docs/api/contracts/upstreamdrift-companion-v1.schema.json"
COMMIT = "1" * 40
OTHER_SHA = "f" * 64
pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def catalog() -> dict[str, Any]:
    return companion_catalog.build_catalog(REPO_ROOT, require_clean=False)


def _committed_sha256(path: str) -> str:
    payload = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{path}"], timeout=30
    )
    return hashlib.sha256(payload).hexdigest()


# --- exported records ------------------------------------------------------


def test_documentation_records_are_exact_immutable_and_routed(
    catalog: dict[str, Any],
) -> None:
    documentation = catalog["documentation"]
    ids = [record["id"] for record in documentation]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert documentation, "the documentation inventory must no longer be empty"
    commit = catalog["source"]["commit"]
    for record in documentation:
        assert record["freshness"] in {
            "current",
            "review_required",
            "stale",
            "unknown",
            "missing",
        }
        if record["status"] == "missing":
            assert record["url"] is None and record["reason"]
            continue
        path = record["source_path"]
        assert record["source_commit"] == commit
        assert record["url"] == (
            f"https://github.com/D-sorganization/UpstreamDrift/blob/{commit}/{path}"
        )
        assert "/blob/main/" not in record["url"]
        assert (REPO_ROOT / path).is_file()
        if record["freshness"] == "current":
            assert record["reason"] is None
            assert record["reviewed_sha256"] == record["source_sha256"]
            assert record["review_due"] >= record["last_reviewed"]
        else:
            assert record["reason"]
    # Every workflow documentation path is a governed record (#4026 cards).
    governed = {r["source_path"] for r in documentation if r["source_path"]}
    for workflow in catalog["workflows"]:
        assert set(workflow["documentation_paths"]) <= governed
    # Every public program exposes an explicit route or an empty route state.
    known_ids = set(ids)
    for program in catalog["programs"]:
        routes = program["documentation_ids"]
        assert routes == sorted(set(routes)) and set(routes) <= known_ids
    assert any(program["documentation_ids"] for program in catalog["programs"])


def test_current_documentation_hashes_match_committed_blobs(
    catalog: dict[str, Any],
) -> None:
    """A recorded review binds to the committed bytes, not the working tree."""
    current = [r for r in catalog["documentation"] if r["freshness"] == "current"]
    assert current, "at least one governed document must be reviewed and current"
    for record in current:
        path = REPO_ROOT / record["source_path"]
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "cat-file",
                "-e",
                f"HEAD:{record['source_path']}",
            ],
            check=False,
            capture_output=True,
            timeout=30,
        )
        expected = (
            _committed_sha256(record["source_path"])
            if tracked.returncode == 0
            and subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPO_ROOT),
                    "diff",
                    "--quiet",
                    "HEAD",
                    "--",
                    record["source_path"],
                ],
                check=False,
                timeout=30,
            ).returncode
            == 0
            else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        assert record["source_sha256"] == expected


def test_engine_capabilities_are_evidence_bounded(catalog: dict[str, Any]) -> None:
    engines = catalog["engines"]
    assert [engine["id"] for engine in engines] == sorted(
        companion_catalog._ENGINE_TIERS
    )
    qualified = 0
    for engine in engines:
        assert engine["support_tier"] == companion_catalog._ENGINE_TIERS[engine["id"]]
        assert engine["scientific_qualification"]["state"] == "unqualified"
        assert engine["runtime_availability"]["state"] in {
            "available",
            "conditional",
            "unavailable",
        }
        assert engine["documentation_ids"]
        doc_ids = {record["id"] for record in catalog["documentation"]}
        assert set(engine["documentation_ids"]) <= doc_ids
        for capability in engine["capabilities"]:
            if capability["evidence_state"] == "qualified":
                qualified += 1
                assert capability["evidence"] and capability["reason"] is None
                for item in capability["evidence"]:
                    assert item["source_commit"] == catalog["source"]["commit"]
                    assert (REPO_ROOT / item["path"]).is_file()
                    assert item["selector"].startswith(f"{item['path']}::")
                    name = item["selector"].rsplit("::", 1)[1]
                    assert f"def {name}(" in (REPO_ROOT / item["path"]).read_text(
                        "utf-8"
                    )
                    assert item["gate"].startswith(".github/workflows/")
                    assert (REPO_ROOT / item["gate"]).is_file()
            else:
                assert capability["evidence"] == [] and capability["reason"]
    assert qualified == catalog["summary"]["qualified_engine_capability_records"]
    assert qualified < catalog["summary"]["engine_capability_records"]


def test_publication_blockers_and_known_gaps_are_derived(
    catalog: dict[str, Any],
) -> None:
    assert catalog["publication"]["state"] == "draft"
    undocumented = sum(
        not p["hidden"] and not p["documentation_ids"] for p in catalog["programs"]
    )
    assert catalog["publication"]["blockers"] == (
        companion_evidence.publication_blockers(
            documentation=catalog["documentation"],
            engines=catalog["engines"],
            known_gaps=catalog["known_gaps"],
            undocumented_visible_programs=undocumented,
        )
    )
    assert companion_evidence.SCREENSHOT_BLOCKER in catalog["publication"]["blockers"]
    gaps = catalog["known_gaps"]
    assert [gap["id"] for gap in gaps] == sorted(gap["id"] for gap in gaps)
    for gap in gaps:
        assert gap["issue"] > 0
        if gap["summary_metric"]:
            assert catalog["summary"][gap["summary_metric"]] > 0
    registries = {record["id"]: record["path"] for record in catalog["registries"]}
    assert (
        registries["documentation"] == "scripts/config/companion_documentation.v1.json"
    )
    assert registries["capability_evidence"] == (
        "scripts/config/companion_capability_evidence.v1.json"
    )


def test_generated_provider_page_is_fresh_and_deterministic() -> None:
    rendered = companion_evidence.render_from_repository(REPO_ROOT)
    committed = (REPO_ROOT / companion_evidence.GENERATED_DOC_PATH).read_text("utf-8")
    assert committed == rendered, (
        "docs/engines/engine_capability_evidence.md is stale; regenerate with "
        "python3 -m scripts.companion_evidence render-docs"
    )
    assert rendered == companion_evidence.render_from_repository(REPO_ROOT)
    # The page is registry-derived only and never references its own commit.
    for record in companion_catalog.build_catalog(REPO_ROOT, require_clean=False)[
        "documentation"
    ]:
        assert record["source_commit"] not in rendered
        if record["source_sha256"]:
            assert record["source_sha256"] not in rendered
    assert "tolerance" not in rendered.split("## Engines")[1].lower()


# --- fail-closed registry contracts ----------------------------------------


def _context(files: dict[str, bytes], *, commit_date: str = "2026-09-17"):
    return companion_evidence.SourceContext(
        commit=COMMIT, commit_date=commit_date, read_input=files.get
    )


def _doc(**overrides: Any) -> dict[str, Any]:
    record = {
        "id": "guide",
        "title": "Guide",
        "status": "active",
        "source_path": "docs/guide.md",
        "audiences": ["user"],
        "topics": ["guide"],
        "program_ids": ["prog"],
        "engine_ids": ["mujoco"],
        "owner": "maintainers",
        "last_reviewed": "2026-09-01",
        "review_due": "2026-12-01",
        "reviewed_sha256": hashlib.sha256(b"guide").hexdigest(),
        "reason": None,
    }
    record.update(overrides)
    return record


def _doc_payload(*records: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            "registry_id": companion_evidence.DOCUMENTATION_REGISTRY_ID,
            "version": "1.0.0",
            "documentation": list(records),
        }
    ).encode("utf-8")


FILES = {"docs/guide.md": b"guide", "docs/other.md": b"other"}


def _parse_docs(*records: dict[str, Any], files=FILES, **kwargs: Any):
    return companion_evidence.parse_documentation_registry(
        _doc_payload(*records),
        context=_context(files, **kwargs),
        program_ids={"prog"},
        engine_ids={"mujoco"},
    )


def test_documentation_freshness_is_derived_from_exact_facts() -> None:
    [current] = _parse_docs(_doc())
    assert current["freshness"] == "current" and current["reason"] is None
    assert current["url"].endswith(f"/blob/{COMMIT}/docs/guide.md")
    [changed] = _parse_docs(_doc(reviewed_sha256=OTHER_SHA))
    assert changed["freshness"] == "review_required" and changed["reason"]
    [stale] = _parse_docs(_doc(), commit_date="2027-01-01")
    assert stale["freshness"] == "stale" and "due" in stale["reason"]
    [unknown] = _parse_docs(
        _doc(last_reviewed=None, review_due=None, reviewed_sha256=None)
    )
    assert unknown["freshness"] == "unknown" and unknown["reason"]
    [missing] = _parse_docs(
        _doc(
            status="missing",
            source_path=None,
            last_reviewed=None,
            review_due=None,
            reviewed_sha256=None,
            reason="No user guide exists yet.",
        )
    )
    assert missing["freshness"] == "missing" and missing["url"] is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"review_due": "2026-08-01"}, "precedes last_reviewed"),
        ({"last_reviewed": "2026-10-01", "review_due": "2026-12-01"}, "later than"),
        ({"reviewed_sha256": "not-a-hash"}, "exact hash"),
        ({"last_reviewed": None, "reviewed_sha256": None}, "require last_reviewed"),
        ({"program_ids": ["ghost"]}, "unknown ids"),
        ({"engine_ids": ["chrono"]}, "unknown ids"),
        ({"source_path": "docs/absent.md"}, "not a tracked file"),
        ({"source_path": "../outside.md"}, "contained repo-relative"),
        ({"status": "deprecated"}, "needs a reason"),
        ({"reason": "custom text"}, "derives its reason"),
        ({"audiences": ["everyone"]}, "audiences"),
        ({"status": "missing", "reason": "gone"}, "cannot declare a source"),
        (
            {"url": "https://github.com/D-sorganization/UpstreamDrift/blob/main/x.md"},
            "unknown or missing keys",
        ),
        ({"freshness": "current"}, "unknown or missing keys"),
    ],
)
def test_documentation_registry_rejects_bad_assertions(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(companion_evidence.EvidenceContractError, match=message):
        _parse_docs(_doc(**overrides))


def test_documentation_registry_rejects_duplicates_and_ungoverned_paths() -> None:
    with pytest.raises(companion_evidence.EvidenceContractError, match="duplicate"):
        _parse_docs(_doc(), _doc())
    with pytest.raises(companion_evidence.EvidenceContractError, match="twice"):
        _parse_docs(_doc(), _doc(id="guide-2"))
    with pytest.raises(companion_evidence.EvidenceContractError, match="no governed"):
        companion_evidence.parse_documentation_registry(
            _doc_payload(_doc()),
            context=_context(FILES),
            program_ids={"prog"},
            engine_ids={"mujoco"},
            required_paths={"docs/other.md"},
        )


TEST_MODULE = (
    b"class TestX:\n    def test_y(self):\n        pass\n\ndef test_z():\n    pass\n"
)
ENGINE_FILES = {
    "tests/test_engine.py": TEST_MODULE,
    ".github/workflows/ci-standard.yml": b"name: ci\n",
    "docs/report.json": b"{}",
}
ENGINE_STUBS = [
    {
        "id": "mujoco",
        "name": "MuJoCo",
        "support_tier": "supported",
        "scientific_qualification": {
            "state": "unqualified",
            "scope": "software only",
            "limitations": ["none claimed"],
        },
    }
]


def _capability(**overrides: Any) -> dict[str, Any]:
    record = {
        "id": "forward",
        "title": "Forward dynamics",
        "evidence_state": "qualified",
        "evidence": [
            {
                "kind": "test",
                "path": "tests/test_engine.py",
                "selector": "tests/test_engine.py::TestX::test_y",
                "gate": ".github/workflows/ci-standard.yml",
            }
        ],
        "reason": None,
        "limitations": ["reference model only"],
    }
    record.update(overrides)
    return record


def _engine(**overrides: Any) -> dict[str, Any]:
    record = {
        "id": "mujoco",
        "runtime_availability": {"state": "available", "reason": None},
        "documentation_ids": ["guide"],
        "capabilities": [_capability()],
    }
    record.update(overrides)
    return record


def _parse_engines(
    *engines: dict[str, Any],
    known_gaps: list[dict[str, Any]] | None = None,
    summary: dict[str, int] | None = None,
):
    payload = json.dumps(
        {
            "registry_id": companion_evidence.CAPABILITY_REGISTRY_ID,
            "version": "1.0.0",
            "engines": list(engines),
            "known_gaps": known_gaps or [],
        }
    ).encode("utf-8")
    return companion_evidence.parse_capability_registry(
        payload,
        context=_context(ENGINE_FILES),
        engines=ENGINE_STUBS,
        documentation_ids={"guide"},
        summary=summary or {"single_source_program_records": 3},
    )


def test_capability_registry_accepts_exact_evidence_and_artifacts() -> None:
    engines, gaps = _parse_engines(
        _engine(
            capabilities=[
                _capability(),
                _capability(
                    id="report",
                    evidence=[
                        {
                            "kind": "artifact",
                            "path": "docs/report.json",
                            "selector": None,
                            "gate": None,
                        }
                    ],
                ),
                _capability(
                    id="parity",
                    evidence_state="unqualified",
                    evidence=[],
                    reason="No gate exists.",
                ),
            ]
        ),
        known_gaps=[
            {
                "id": "divergence",
                "issue": 8853,
                "scope": "catalog",
                "summary": "Registries disagree.",
                "summary_metric": "single_source_program_records",
            }
        ],
    )
    [engine] = engines
    assert engine["support_tier"] == "supported" and engine["name"] == "MuJoCo"
    assert engine["scientific_qualification"]["state"] == "unqualified"
    by_id = {c["id"]: c for c in engine["capabilities"]}
    assert (
        by_id["forward"]["evidence"][0]["sha256"]
        == hashlib.sha256(TEST_MODULE).hexdigest()
    )
    assert by_id["forward"]["evidence"][0]["source_commit"] == COMMIT
    assert by_id["report"]["evidence"][0]["gate"] is None
    assert by_id["parity"]["evidence_state"] == "unqualified"
    assert gaps[0]["issue"] == 8853


@pytest.mark.parametrize(
    ("capability", "message"),
    [
        ({"evidence": []}, "need evidence"),
        ({"reason": "but qualified"}, "need evidence and no reason"),
        ({"evidence_state": "unqualified", "reason": "x"}, "cannot carry evidence"),
        ({"evidence_state": "unqualified", "evidence": [], "reason": None}, "reason"),
        ({"evidence_state": "validated", "evidence": []}, "unsupported evidence state"),
        (
            {
                "evidence": [
                    {
                        "kind": "test",
                        "path": "tests/test_engine.py",
                        "selector": "tests/test_engine.py::TestX::test_missing",
                        "gate": ".github/workflows/ci-standard.yml",
                    }
                ]
            },
            "does not exist",
        ),
        (
            {
                "evidence": [
                    {
                        "kind": "test",
                        "path": "tests/test_engine.py",
                        "selector": "other.py::test_z",
                        "gate": ".github/workflows/ci-standard.yml",
                    }
                ]
            },
            "must be",
        ),
        (
            {
                "evidence": [
                    {
                        "kind": "test",
                        "path": "tests/test_engine.py",
                        "selector": "tests/test_engine.py::test_z",
                        "gate": "scripts/run.sh",
                    }
                ]
            },
            "workflow under",
        ),
        (
            {
                "evidence": [
                    {
                        "kind": "test",
                        "path": "tests/test_engine.py",
                        "selector": "tests/test_engine.py::test_z",
                        "gate": ".github/workflows/nightly.yml",
                    }
                ]
            },
            "not tracked",
        ),
        (
            {
                "evidence": [
                    {
                        "kind": "test",
                        "path": "tests/absent.py",
                        "selector": "tests/absent.py::test_z",
                        "gate": ".github/workflows/ci-standard.yml",
                    }
                ]
            },
            "not a tracked file",
        ),
        (
            {
                "evidence": [
                    {
                        "kind": "artifact",
                        "path": "docs/report.json",
                        "selector": "x::y",
                        "gate": None,
                    }
                ]
            },
            "artifact evidence",
        ),
        ({"tolerances": {"position": 1e-6}}, "unknown or missing keys"),
        (
            {"scientific_qualification": {"state": "qualified"}},
            "unknown or missing keys",
        ),
    ],
)
def test_capability_registry_rejects_unsupported_promotion(
    capability: dict[str, Any], message: str
) -> None:
    with pytest.raises(companion_evidence.EvidenceContractError, match=message):
        _parse_engines(_engine(capabilities=[_capability(**capability)]))


@pytest.mark.parametrize(
    ("engine", "message"),
    [
        ({"id": "drake"}, "contradict"),
        ({"support_tier": "supported"}, "unknown or missing keys"),
        ({"name": "MuJoCo"}, "unknown or missing keys"),
        (
            {"runtime_availability": {"state": "available", "reason": "x"}},
            "must be null",
        ),
        ({"runtime_availability": {"state": "conditional", "reason": None}}, "reason"),
        ({"documentation_ids": []}, "must not be empty"),
        ({"documentation_ids": ["ghost"]}, "unknown ids"),
        ({"capabilities": [_capability(), _capability()]}, "duplicate capability"),
    ],
)
def test_capability_registry_rejects_contradictory_engine_facts(
    engine: dict[str, Any], message: str
) -> None:
    with pytest.raises(companion_evidence.EvidenceContractError, match=message):
        _parse_engines(_engine(**engine))


def test_capability_registry_requires_every_catalog_engine() -> None:
    with pytest.raises(companion_evidence.EvidenceContractError, match="contradict"):
        _parse_engines(_engine(), _engine())


@pytest.mark.parametrize(
    ("gap", "message"),
    [
        ({"issue": 0}, "positive"),
        ({"scope": "vibes"}, "unsupported scope"),
        ({"summary_metric": "not_exported"}, "not exported"),
        ({"summary_metric": "zero"}, "no longer observed"),
    ],
)
def test_known_gaps_need_owning_issues_and_live_observations(
    gap: dict[str, Any], message: str
) -> None:
    record = {
        "id": "gap",
        "issue": 8853,
        "scope": "catalog",
        "summary": "Registries disagree.",
        "summary_metric": None,
    }
    record.update(gap)
    with pytest.raises(companion_evidence.EvidenceContractError, match=message):
        _parse_engines(
            _engine(),
            known_gaps=[record],
            summary={"single_source_program_records": 3, "zero": 0},
        )


# --- schema negatives --------------------------------------------------------


def test_schema_rejects_mutable_links_and_unqualified_promotion(
    catalog: dict[str, Any],
) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(catalog)
    current = next(r for r in catalog["documentation"] if r["freshness"] == "current")
    qualified_engine = next(
        e
        for e in catalog["engines"]
        if any(c["evidence_state"] == "qualified" for c in e["capabilities"])
    )

    def broken(mutate) -> dict[str, Any]:
        copy_ = copy.deepcopy(catalog)
        mutate(copy_)
        return copy_

    def mutable_url(c):
        record = next(r for r in c["documentation"] if r["id"] == current["id"])
        record["url"] = (
            "https://github.com/D-sorganization/UpstreamDrift/blob/main/"
            + record["source_path"]
        )

    def reason_on_current(c):
        next(r for r in c["documentation"] if r["id"] == current["id"])["reason"] = "x"

    def qualified_without_evidence(c):
        engine = next(e for e in c["engines"] if e["id"] == qualified_engine["id"])
        cap = next(
            x for x in engine["capabilities"] if x["evidence_state"] == "qualified"
        )
        cap["evidence"] = []

    def promoted_engine(c):
        c["engines"][0]["scientific_qualification"]["state"] = "approved"

    def missing_route(c):
        c["programs"][0].pop("documentation_ids")

    def gap_without_issue(c):
        c["known_gaps"].append(
            {
                "id": "x",
                "issue": None,
                "scope": "catalog",
                "summary": "s",
                "summary_metric": None,
            }
        )

    for mutate in (
        mutable_url,
        reason_on_current,
        qualified_without_evidence,
        promoted_engine,
        missing_route,
        gap_without_issue,
    ):
        assert list(validator.iter_errors(broken(mutate))), mutate.__name__


def test_evidence_module_imports_nothing_from_src() -> None:
    text = (REPO_ROOT / "scripts/companion_evidence.py").read_text(encoding="utf-8")
    assert "from src" not in text and "import src" not in text
