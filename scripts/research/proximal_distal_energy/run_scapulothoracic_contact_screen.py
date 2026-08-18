"""Generate the governed scapulothoracic contact-screen artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scapulothoracic_contact_screen import run_scapulothoracic_contact_screen

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
JSON_PATH = ARTICLE / "data/scapulothoracic_contact_screen.json"
NPZ_PATH = ARTICLE / "data/scapulothoracic_contact_screen.npz"


def main() -> None:
    """Run the predeclared screen and write deterministic JSON/NPZ evidence."""
    record, arrays = run_scapulothoracic_contact_screen()
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez(NPZ_PATH, **arrays)
    print(JSON_PATH)
    print(NPZ_PATH)


if __name__ == "__main__":
    main()
