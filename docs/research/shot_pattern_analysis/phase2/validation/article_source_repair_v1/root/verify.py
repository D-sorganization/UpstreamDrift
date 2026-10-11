import sys
import collections
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path("/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift")
P = ROOT / "docs/research/shot_pattern_analysis/phase2"
ARTICLE = P.parent / "shot_pattern_analysis.tex"


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


trace_path = P / "validation/selected_headline_traceability_v1/review.json"
trace = json.loads(trace_path.read_text())
ref_path = P / "validation/protocol_name_recovery_v6/expected_numeric_hashes.json"
ref = json.loads(ref_path.read_text())
bindings = {P / name: digest for name, digest in trace["source_sha256"].items()}
bindings.update(
    {P / "supplement_report_v1" / name: digest for name, digest in ref.items()}
)
bindings[trace_path] = h(trace_path)
bindings[ref_path] = h(ref_path)
for p, d in bindings.items():
    if h(p) != d:
        raise ValueError(f"changed input {p}")
cells = {f"C{i:02}": c for i, c in enumerate(trace["cells"], 1)}
rows = {
    (f"C{i:02}", r["pattern"]): r
    for i, c in enumerate(trace["cells"], 1)
    for r in trace["rows"]
    if r["cell"] == c["cell"]
}
comparisons = json.loads(
    (P / "supplement_report_v1/REPORT_SUPPLEMENT_COMPARISONS.json").read_text()
)["comparisons"]
proto = {
    "P": "original_primary/1",
    "OC": "original_control/1",
    "WP": "pw-primary-face-bounds/1",
    "WC": "pw-control-face-bounds/1",
    "DC": "driver-control-speed-bounds/1",
}
effects: dict[str, Any] = {}
expected_e: dict[str, Any] = {}
for club in ["driver", "seven_iron", "pitching_wedge"]:
    for profile in ["shaft_only", "combined2"]:
        for kind, a, b, pb in [
            (
                "shaft_rotation_sd",
                f"{club}_lean0_curve1_rotation1_{profile}",
                f"{club}_lean0_curve1_rotation2_{profile}",
                "OC",
            ),
            (
                "curve_strength",
                f"{club}_lean0_curve1_rotation2_{profile}",
                f"{club}_lean0_curve2_rotation2_{profile}",
                "WP" if club == "pitching_wedge" else "OC",
            ),
        ]:
            key = f"E{len(effects) + 1:02}"
            ids = [(proto["OC"], a), (proto[pb], b)]
            matches = [
                x
                for x in comparisons
                if x["comparison_kind"] == kind
                and [(s["protocol_id"], s["cell"]) for s in x["sources"]] == ids
            ]
            if len(matches) != 1:
                raise ValueError("ambiguous interaction")
            effects[key] = matches[0]
            expected_e[key] = ids
roots = {
    "OC": P / "control_final_v1",
    "WP": P / "supplement_final_v1/pw_primary_face",
    "WC": P / "supplement_final_v1/pw_control_face",
    "DC": P / "supplement_final_v1/driver_control_speed",
}
nominals: dict[str, Any] = {}
for key, club, pa, pb in [
    ("F1", "driver", "OC", "DC"),
    ("F2", "pitching_wedge", "WP", "WC"),
]:
    ids = [
        (proto[pa], f"{club}_lean0_curve2_rotation2_combined2"),
        (proto[pb], f"{club}_lean5_curve2_rotation2_combined2"),
    ]
    found = [
        x
        for x in comparisons
        if x["comparison_kind"] == "across_lean"
        and [(s["protocol_id"], s["cell"]) for s in x["sources"]] == ids
    ]
    if len(found) != 1:
        raise ValueError("ambiguous control")
    effects[key] = found[0]
    for suffix, pc, source in zip("AB", [pa, pb], found[0]["sources"], strict=True):
        root = roots[pc] / source["cell"]
        mp = root / "manifest.json"
        sp = root / "summary.json"
        if h(mp) != source["source_cell_manifest_sha256"]:
            raise ValueError("manifest mismatch")
        manifest = json.loads(mp.read_text())
        bindings[mp] = h(mp)
        bindings[sp] = manifest["files_sha256"]["summary.json"]
        if h(sp) != bindings[sp]:
            raise ValueError("summary mismatch")
        nominals[key + suffix] = json.loads(sp.read_text())["nominals"]


def number(display: str, want: Any) -> None:
    a = Decimal(display)
    b = Decimal(str(want))
    if not a.is_finite() or not b.is_finite():
        raise ValueError(f"nonfinite Decimal input: display={display!r}, want={want!r}")
    exp = a.as_tuple().exponent
    if not isinstance(exp, int):
        raise ValueError(f"non-integer Decimal exponent: {exp!r}")
    if a == 0:
        if b != 0:
            raise ValueError(("rounded-to-zero", display, want))
    elif abs(a - b) > Decimal("0.50000001") * (Decimal(10) ** exp) + abs(b) * Decimal(
        "1e-14"
    ):
        raise ValueError(("display precision mismatch", display, want))


def _verify_identity_row(line: str, f: list[str], seen: Any) -> bool:
    if re.match(r"E\d\d & (Twist|Curve)", line):
        key = f[0]
        ids = []
        for item in f[2:4]:
            ck, pc = item.split("/")
            ids.append((proto[pc], cells[ck]["cell"]))
        if ids != expected_e[key] or key in seen["keys"]:
            raise ValueError("interaction identity")
        seen["keys"].add(key)
        return True
    if re.match(r"C\d\d & (Driver|7-Iron|Pitching Wedge)", line):
        key = f[0]
        c = cells[key]
        cfg = c["config"]
        club = {
            "driver": "Driver",
            "seven_iron": "7-Iron",
            "pitching_wedge": "Pitching Wedge",
        }[cfg["club"]["club_id"]]
        expected_fields = [
            key,
            club,
            str(int(cfg["curve_scale"])),
            str(int(cfg["noise"]["standard_deviations"][0])),
            "S" if c["cell"].endswith("shaft_only") else "C",
            next(k for k, v in proto.items() if v == c["source_protocol"]),
        ]
        if f != expected_fields or key in seen["conditions"]:
            raise ValueError("condition key metadata")
        seen["conditions"].add(key)
        return True
    return False


def _synthesis_row(f: list[str]) -> tuple[str, Any, list[Any]]:
    club = {
        "Driver": "driver",
        "7-Iron": "seven_iron",
        "Pitching Wedge": "pitching_wedge",
    }[f[0]]
    pat = f[1]
    items = [
        r["paired_sg_vs_straight"]
        for r in rows.values()
        if r["club"] == club and r["pattern"] == pat
    ]
    counts = [
        sum(x["lower_95"] > 0 for x in items),
        sum(x["upper_95"] < 0 for x in items),
        sum(x["lower_95"] <= 0 <= x["upper_95"] for x in items),
    ]
    if f[4] != "/".join(map(str, counts)):
        raise ValueError("interval direction counts")
    group = "synthesis"
    identity = (club, pat)
    want = [
        min(x["estimate"] for x in items),
        max(x["estimate"] for x in items),
    ]
    return group, identity, want


def _numeric_row(line: str, f: list[str]) -> tuple[str, Any, list[Any]] | None:
    if re.match(r"C\d\d & (Straight|Draw|Fade)", line):
        key, pat = f[:2]
        r = rows[key, pat]
        group = "geometry" if len(f) == 7 else "paired"
        identity = (key, pat)
        if group == "geometry":
            if int(f[2]) != r["supported_n"]:
                raise ValueError("endpoint count")
            want = [
                r[k]
                for k in [
                    "lateral_sd_m",
                    "downrange_sd_m",
                    "target_rmse_m",
                    "mean_sg_strokes",
                ]
            ]
        else:
            if int(f[2]) != r["scoring_paired_n"]:
                raise ValueError("scoring count")
            want = [
                r["paired_sg_vs_straight"][k]
                for k in ["estimate", "lower_95", "upper_95"]
            ]
    elif re.match(r"[EF]\d\d? & (Draw|Fade)", line):
        key, pat = f[:2]
        r = effects[key]
        group = "effect"
        identity = (key, pat)
        if int(f[2]) != r["common_supported_n"]:
            raise ValueError("common six count")
        want = [
            r["course_scoring_effect_modification"][pat][k]
            for k in ["estimate", "lower_95", "upper_95"]
        ]
    elif re.match(r"F[12][AB] &", line):
        key, pat = f[:2]
        group = "nominal"
        identity = (key, pat)
        r = nominals[key][pat]
        want = [r[k] for k in ["face_deg", "path_deg", "speed_mps", "residual_m"]]
    elif re.match(r"(Driver|7-Iron|Pitching Wedge) & (Draw|Fade)", line):
        return _synthesis_row(f)
    else:
        return None
    return group, identity, want


def _verify_coverage(seen: Any) -> None:
    expected = {
        "geometry": set(rows),
        "paired": {k for k in rows if k[1] != "Straight"},
        "effect": {(k, p) for k in effects for p in ["Draw", "Fade"]},
        "nominal": {(k, p) for k in nominals for p in ["Straight", "Draw", "Fade"]},
        "keys": set(expected_e),
        "conditions": set(cells),
        "synthesis": {
            (c, p)
            for c in ["driver", "seven_iron", "pitching_wedge"]
            for p in ["Draw", "Fade"]
        },
    }
    if dict(seen) != expected:
        raise ValueError("incomplete identity coverage")


def verify(s: str) -> dict[str, Any]:
    seen: collections.defaultdict[str, set[Any]] = collections.defaultdict(set)
    n = 0
    for line in s.splitlines():
        f = [v.strip() for v in line.rstrip("\\ ").split("&")]
        nums = re.findall(r"\\numtext\{([^}]+)\}", line)
        if _verify_identity_row(line, f, seen):
            continue
        row = _numeric_row(line, f)
        if row is None:
            continue
        group, identity, want = row
        if identity in seen[group]:
            raise ValueError("duplicate row")
        seen[group].add(identity)
        if len(nums) != len(want):
            raise ValueError("field count")
        for a, b in zip(nums, want, strict=True):
            number(a, b)
            n += 1
    _verify_coverage(seen)
    residual = re.search(
        r"maximum residual among the 72 selected nominal solutions is \\numtext\{([^}]+)\}",
        s,
    )
    if residual is None:
        raise ValueError("residual missing")
    number(
        residual.group(1),
        max(
            nom["residual_m"] for c in cells.values() for nom in c["nominals"].values()
        ),
    )
    n += 1
    return {"numeric_fields": n, "row_counts": {k: len(v) for k, v in seen.items()}}


s = ARTICLE.read_text()
result = verify(s)
mutations = {
    "fine_precision": s.replace("\\numtext{-0.395737}", "\\numtext{-0.395730}", 1),
    "wrong_population": s.replace(
        "C01 & Draw & 10000 & \\numtext{+7.461556e-05}",
        "C01 & Draw & 9999 & \\numtext{+7.461556e-05}",
        1,
    ),
    "wrong_protocol": s.replace(
        "E01 & Twist SD 1 to 2 & C01/OC", "E01 & Twist SD 1 to 2 & C01/P", 1
    ),
    "duplicate_row": s
    + "\n"
    + next(x for x in s.splitlines() if x.startswith("C01 & Straight & 10000")),
}
for name, mutant in mutations.items():
    if mutant == s:
        raise ValueError("ineffective negative control")
    try:
        verify(mutant)
    except ValueError:
        continue
    raise ValueError("negative control accepted " + name)
for p, d in bindings.items():
    if h(p) != d:
        raise ValueError("input changed during review")
result.update(
    article_sha256=h(ARTICLE),
    script_sha256=h(Path(__file__)),
    input_sha256={str(p.relative_to(P)): d for p, d in bindings.items()},
    negative_controls_rejected=list(mutations),
    compiled=False,
    visual_review=False,
    full_publication_accepted=False,
)
Path("/tmp/p2-article-root-review/review.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
sys.stdout.write(
    str(json.dumps({k: v for k, v in result.items() if k != "input_sha256"}, indent=2))
    + "\n"
)
