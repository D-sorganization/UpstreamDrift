---
title: Project Map
tile_id: project_map
status: complete
---

# Project Map

## Purpose

Project Map opens `docs/architecture/PROJECT_MAP.md`, the suite's single
comprehensive reference for every feature, module and tool in UpstreamDrift.
Its stated purpose is full visibility into what the platform can do, including
features that are not yet exposed as launcher tiles.

The tile is registered with `type: document`, `feature_id:
docs.document_library`, `launcher.category: documentation`, and
`path: docs/architecture/PROJECT_MAP.md`, so it resolves as an ordinary
in-repository file rather than as a program. Declared capabilities:
`documentation`, `feature_discovery`, `project_overview`.

## Inputs

| Input | Description | Unit or values |
| --- | --- | --- |
| Document path | fixed by the registry to `docs/architecture/PROJECT_MAP.md` | repository-relative path |
| Host platform | selects the open mechanism | Windows, macOS, or other |

There are no user-supplied parameters. Nothing about the document is
configurable from the tile.

## Outputs

| Output | Description |
| --- | --- |
| Rendered document | the Project Map opened for reading, in a launcher document view or in the system default viewer |
| Exit status | non-zero from the proxy runner when the file is missing or cannot be opened |

The document itself carries, as of its version 2.1.1 header dated 2026-06-10,
sixteen sections: launcher tiles, physics engines, the model gait and
locomotion system, the robotics module, learning and AI, research modules,
deployment and real-time, Unreal Engine integration, the shared analysis
library, tools and utilities, visualization and plotting, API and web UI,
examples and tutorials, hidden or unexposed features, deprecated or archived
code, and an operational status and gap inventory. Section 16 is the gap
inventory of started-but-unfinished seams, each tracked by a GitHub issue.

## Method

The tile is a document, not an application, so "launching" it means opening a
file. `src/launchers/document_proxy.py` is the approved executable under the
`secure_subprocess` whitelist for opening non-executable documentation files.
It resolves the path, exits with status 1 when the file does not exist, and
otherwise hands the file to the platform viewer: `os.startfile` on Windows,
`open` on macOS, `xdg-open` elsewhere. A failure to open is logged and exits
non-zero.

Section 1 of the document itself states that the visible launcher tiles are
defined in `src/config/launcher_manifest.json` and `src/config/models.yaml`,
which is the same registry pair that declares this tile.

## Limitations

- It is a static Markdown file. There is no search, no filtering, no
  cross-linking into the code and no live query of the registry.
- It can go stale. The document carries its own version and date in its
  header; nothing regenerates it from the registry, so a tile added or renamed
  in `models.yaml` does not update the Project Map.
- No web surface. The registry marks `web.mode: unavailable` with the reason
  "Documentation file; no dedicated web page yet", so the tile has no web page
  even though it lists `web` among its surfaces.
- Rendering is the viewer's business. Opened through `document_proxy.py`, the
  Markdown appears in whatever the operating system associates with `.md`,
  which may be a plain text editor with no rendering at all.
- The document counts eleven launcher tiles in its section 1. The registry now
  declares considerably more, so treat that count as a snapshot rather than a
  current inventory.

## See Also
- [Project Map document](../architecture/PROJECT_MAP.md)
- [System overview](../architecture/system_overview.md)
- [Feature roadmap](../architecture/feature_roadmap.md)
- [User manual](../user_guide/user_manual.md)
