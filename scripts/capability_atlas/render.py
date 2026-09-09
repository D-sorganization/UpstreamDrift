"""Accessible SVG/HTML reference plus standard Mermaid diagrams from one graph."""

from __future__ import annotations

import html
import textwrap
from collections import defaultdict
from urllib.parse import quote

from .model import Graph

REPO_URL = "https://github.com/D-sorganization/UpstreamDrift"


def source_link(path: str, label: str = "Source") -> str:
    repo_url = REPO_URL
    if path.startswith("vendor/ud-tools/"):
        repo_url = "https://github.com/D-sorganization/Tools"
        path = path.removeprefix("vendor/ud-tools/")
    url = f"{repo_url}/blob/main/{quote(path, safe='/')}"
    return f'<a href="{url}">{html.escape(label)}</a>'


def view_graph(graph: Graph, view: str) -> tuple[list[dict], list[dict]]:
    edges = [e for e in graph["edges"] if e["view"] == view]
    ids = {e[key] for e in edges for key in ("source", "target")}
    nodes = [n for n in graph["nodes"] if n["view"] == view or n["id"] in ids]
    return nodes, edges


def mermaid(graph: Graph, view: str) -> str:
    """Portable Mermaid flowchart with accessible description and artifact labels."""
    nodes, edges = view_graph(graph, view)
    ids = {n["id"]: f"n{i}" for i, n in enumerate(nodes)}
    title = (
        "Capture Workflow" if view == "workflow" else "System Context and Containers"
    )
    lines = [
        "flowchart LR",
        f"  accTitle: {title}",
        "  accDescr: Source-backed connections. File arrows require explicit artifact exchange.",
    ]
    for node in nodes:
        # Mermaid entity escapes keep quotes/brackets from introducing syntax.
        title_text = html.escape(node["title"]).replace('"', "#quot;")
        lines.append(f'  {ids[node["id"]]}["{title_text}"]')
    for edge in edges:
        label = html.escape(edge["artifact"]).replace('"', "#quot;")
        arrow = "-.->" if edge["kind"] == "file" else "-->"
        lines.append(
            f'  {ids[edge["source"]]} {arrow}|"{label}"| {ids[edge["target"]]}'
        )
    return "\n".join(lines) + "\n"


def _positions(nodes: list[dict], edges: list[dict]) -> dict[str, tuple[int, int]]:
    """Layer this small acyclic reference; Mermaid is the editable diagram export."""
    remaining = {n["id"] for n in nodes}
    ranks: dict[str, int] = {}
    while remaining:
        ready = sorted(
            n
            for n in remaining
            if all(e["source"] in ranks for e in edges if e["target"] == n)
        )
        if not ready:
            raise ValueError("Capability view contains a cycle; use a sequence view")
        for node in ready:
            parents = [ranks[e["source"]] for e in edges if e["target"] == node]
            ranks[node] = max(parents, default=-1) + 1
            remaining.remove(node)
    columns: dict[int, list[str]] = defaultdict(list)
    result: dict[str, tuple[int, int]] = {}
    for node, rank in ranks.items():
        columns[rank].append(node)
    for rank in sorted(columns):

        def parent_position(node: str) -> tuple[float, str]:
            parents = [result[e["source"]][1] for e in edges if e["target"] == node]
            return sum(parents) / max(len(parents), 1), node

        for row, node in enumerate(sorted(columns[rank], key=parent_position)):
            result[node] = (35 + rank * 255, 50 + row * 175)
    return result


def svg(graph: Graph, view: str) -> str:
    """Offline reference thumbnail with source links and a full text alternative."""
    nodes, edges = view_graph(graph, view)
    pos = _positions(nodes, edges)
    width = max(x for x, _ in pos.values()) + 245
    height = max(y for _, y in pos.values()) + 155
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="{view}-title {view}-desc">',
        f'<title id="{view}-title">{view.title()} Capability Network</title>',
        f'<desc id="{view}-desc">Connections are detailed in the adjacent table. '
        "Dashed arrows represent file exchange, not automatic execution.</desc>",
        f'<defs><marker id="arrow-{view}" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" '
        'fill="#76b9bf"/></marker></defs>',
    ]
    for index, edge in enumerate(edges, 1):
        x, y = pos[edge["source"]]
        tx, ty = pos[edge["target"]]
        start, end = x + 210, tx - 8
        dash = ' stroke-dasharray="5 4"' if edge["kind"] == "file" else ""
        lines.append(
            f'<path d="M{start},{y + 54} C{start + 30},{y + 54} {end - 30},{ty + 54} '
            f'{end},{ty + 54}" fill="none" stroke="#76b9bf" stroke-width="2" '
            f'marker-end="url(#arrow-{view})"{dash}><title>'
            f"{html.escape(edge['artifact'] + ': ' + edge['constraint'])}</title></path>"
        )
        lines.append(
            f'<text x="{start + 9}" y="{y + 43}" fill="#c3d9df" '
            f'font-size="11">{index}</text>'
        )
    for node in nodes:
        x, y = pos[node["id"]]
        url = f"{REPO_URL}/blob/main/{quote(node['evidence'], safe='/')}"
        lines += [
            f'<a href="{url}" aria-label="{html.escape(node["title"], quote=True)} source">',
            f'<rect x="{x}" y="{y}" width="210" height="110" rx="14" '
            'fill="#152d40" stroke="#436274"/>',
        ]
        for row, label in enumerate(textwrap.wrap(node["title"], width=25)[:3]):
            lines.append(
                f'<text x="{x + 15}" y="{y + 30 + row * 21}" fill="#edf5f7" '
                f'font-size="15">{html.escape(label)}</text>'
            )
        lines.append("</a>")
    return "\n".join(lines + ["</svg>"])


def edge_table(graph: Graph, view: str) -> str:
    nodes = {n["id"]: n["title"] for n in graph["nodes"]}
    lines = [
        '<div class="table-scroll"><table><caption>Capability Connections</caption>',
        '<thead><tr><th scope="col"># / Connection</th><th scope="col">Artifact</th>'
        '<th scope="col">Meaning and Limits</th></tr></thead><tbody>',
    ]
    edges = [e for e in graph["edges"] if e["view"] == view]
    for index, edge in enumerate(edges, 1):
        label = f"{index}. {nodes[edge['source']]} → {nodes[edge['target']]}"
        lines.append(
            f'<tr><th scope="row">{html.escape(label)}</th>'
            f"<td>{html.escape(edge['artifact'])}<br><small>{edge['kind']}</small></td>"
            f"<td>{html.escape(edge['constraint'])} "
            f"{source_link(edge['evidence'])}</td></tr>"
        )
    return "\n".join(lines + ["</tbody></table></div>"])


def catalog(graph: Graph) -> str:
    cards = []
    for feature in graph["features"]:
        title = feature["title"]
        sources = " · ".join(
            source_link(feature[key], label)
            for key, label in (("pyqt", "Desktop"), ("api", "API"), ("web", "Web"))
            if feature.get(key)
        )
        status = feature["status"]
        defaults = {
            "parity": "Registry-declared surface parity.",
            "gap": "One or more surfaces remain incomplete; see the tracking issue.",
            "exempt": "The registry records an exception to surface parity.",
        }
        explanation = feature.get("reason") or feature.get("notes") or defaults[status]
        if feature.get("pending_decision"):
            explanation = "Exemption pending review. " + explanation
        if feature.get("issue"):
            explanation += f" Tracking issue #{feature['issue']}."
        cards.append(
            f'<article data-entry data-status="{status}"><span class="badge {status}">'
            f"{status}</span><h3>{html.escape(title)}</h3><p>{html.escape(explanation)}</p>"
            f"<small>{html.escape(feature['id'])}</small><p>{sources}</p></article>"
        )
    return "\n".join(cards)


def tile_catalog(graph: Graph) -> str:
    lines = []
    for tile in graph["tiles"]:
        lines.append(
            f'<article data-entry data-status="launcher"><span class="badge">'
            f"{html.escape(tile['status'])}</span><h3>{html.escape(tile['title'])}</h3>"
            f"<p>{html.escape(tile['description'])}</p><small>{html.escape(tile['category'])}"
            f" · {html.escape(tile['id'])}</small><p>{source_link(tile['path'])}</p></article>"
        )
    return "\n".join(lines)


def document(graph: Graph) -> str:
    lines = [
        "# Generated Capability and Architecture Map",
        "",
        "<!-- Generated: python3 -m scripts.generate_capability_atlas -->",
        "",
        "[Open the Interactive Reference](../../ui/public/capability-atlas/index.html)",
        "",
        f"**{len(graph['tiles'])} launcher tiles · {len(graph['features'])} feature contracts.**",
        "This catalog follows the existing launcher and parity registries. A registry",
        "status is not a runtime health check or scientific validation.",
        "",
        "## System Context and Containers",
        "",
        "```mermaid",
        mermaid(graph, "system").rstrip(),
        "```",
        "",
        "## Capture Workflow",
        "",
        "```mermaid",
        mermaid(graph, "workflow").rstrip(),
        "```",
        "",
        "Dashed connections exchange files explicitly. Single-view analysis is",
        "2-D; calibrated reconstruction requires multiple views and geometric observability.",
        "",
        "## Feature Surfaces",
        "",
        "| Feature | Registry Status | Source Surfaces |",
        "| --- | --- | --- |",
    ]
    for item in graph["features"]:
        links = [
            f"[{key}]({REPO_URL}/blob/main/{quote(item[key], safe='/')})"
            for key in ("pyqt", "api", "web")
            if item.get(key)
        ]
        title = item["title"].replace("|", "&#124;")
        lines.append(f"| {title} | {item['status']} | {' · '.join(links)} |")
    lines += [
        "",
        "## Regeneration and Evidence",
        "",
        "Semantic connections are declared in `src/config/capability_connections.json`",
        "with artifact names, constraints and source evidence. Workflow labels and",
        "instructions come from `capture_rig.workflow`; feature/launcher inventories",
        "are consumed rather than copied. Initialize the pinned Tools submodule first.",
        "",
        "Run `python3 -m scripts.generate_capability_atlas --check` to check freshness.",
        "`tests/scripts/test_capability_atlas.py` gates deterministic outputs and invalid graphs.",
        "",
        "Standards: [C4](https://c4model.com/introduction),",
        "[Mermaid](https://mermaid.js.org/syntax/flowchart.html).",
        "",
    ]
    return "\n".join(lines)
