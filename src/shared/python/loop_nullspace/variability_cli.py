"""Private-capture entry point for the NSX-97 falsification report.

Reads an ``.npz`` prepared from a **private** anonymous capture directory
(arrays ``states`` (N,n), ``swing_ids`` (N,), ``session_ids`` (N,),
``outcome_jacobian`` (m,n), ``outcome_target`` (m,), optional
``outcomes`` (N,m)) and writes a JSON report plus a hash receipt to paths the
caller chooses (outside the repository).  The receipt holds SHA-256 digests
and sizes only: no file names, no paths.  Running this on a real capture is
the deferred validation; this code makes no capture claim.

    python -m src.shared.python.loop_nullspace.variability_cli \
        --input cap.npz --output report.json --receipt receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .variability_report import FalsificationOptions, falsify_variability
from .variability_split import SessionSplit, SwingSample
from .variability_tnc import DEFAULT_SEED

_CHUNK = 1 << 20


@dataclass(frozen=True)
class FileDigest:
    """SHA-256 and size of one input (position only, never the name)."""

    index: int
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CaptureReceipt:
    """Input provenance receipt: digests only, safe to store privately."""

    files: tuple[FileDigest, ...]
    combined_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [asdict(f) for f in self.files],
            "combined_sha256": self.combined_sha256,
        }


def hash_inputs(paths: Sequence[Path]) -> CaptureReceipt:
    """Hash input files in the given order.

    Raises:
        FileNotFoundError: If any path is not a file.
        ValueError: If ``paths`` is empty.
    """
    if not paths:
        raise ValueError("no input files to hash")
    digests = []
    for i, path in enumerate(paths):
        if not Path(path).is_file():
            raise FileNotFoundError("input file not found")
        h = hashlib.sha256()
        size = 0
        with open(path, "rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
                size += len(chunk)
        digests.append(FileDigest(i, h.hexdigest(), size))
    combined = hashlib.sha256("".join(d.sha256 for d in digests).encode()).hexdigest()
    return CaptureReceipt(tuple(digests), combined)


def _load(path: Path) -> tuple[list[SwingSample], np.ndarray, np.ndarray, Any]:
    with np.load(path, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=float)
        ids, sess = data["swing_ids"], data["session_ids"]
        samples = [
            SwingSample(str(i), str(s), x)
            for i, s, x in zip(ids, sess, states, strict=True)
        ]
        outcomes = data["outcomes"] if "outcomes" in data.files else None
        return samples, data["outcome_jacobian"], data["outcome_target"], outcomes


def main(argv: Sequence[str] | None = None) -> int:
    """Run the falsification and write the report and receipt."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--rotations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    receipt = hash_inputs([args.input])
    samples, jac, target, outcomes = _load(args.input)
    split = SessionSplit.from_sessions(
        (s.session_id for s in samples), args.test_fraction, args.seed
    )
    test_ids = [s for s in samples if s.session_id in split.test_sessions]
    if outcomes is not None:
        keep = {s.swing_id for s in test_ids}
        outcomes = np.asarray(outcomes)[
            [i for i, s in enumerate(samples) if s.swing_id in keep]
        ]
    result = falsify_variability(
        samples,
        split,
        jac,
        target,
        test_outcomes=outcomes,
        options=FalsificationOptions(
            n_permutations=args.permutations,
            n_rotations=args.rotations,
            seed=args.seed,
        ),
    )
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    args.receipt.write_text(json.dumps(receipt.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
