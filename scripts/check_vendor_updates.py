"""Check whether vendored submodule dependencies are up to date.

This script compares the commit SHAs pinned in .gitmodules / git-submodule
metadata against the latest commit on the upstream tracking branch (typically
``main``).  It exits with a non-zero status when any submodule is behind so
that CI pipelines can surface the information as a build notice.

Professional Rationale
----------------------
The "check for updates" question for vendored shared tools has a well-known
answer in production-grade software:

1. **Git submodules** (our approach) are the correct mechanism for pinning a
   specific, reproducible snapshot of an external dependency.  The submodule
   pointer commit SHA acts as a "lock file" guaranteeing reproducibility.

2. **In-repo update script** (this file) lets every developer and every CI run
   see at a glance whether the vendor snapshot is stale without requiring a
   full ``git submodule update --remote`` which mutates state.

3. **Automated CI job** (`.github/workflows/vendor-freshness.yml`) runs this
   script nightly and opens a bumping PR automatically when the upstream
   advances.  This prevents the submodule from silently drifting far from
   upstream.

Usage
-----
From the UpstreamDrift repository root::

    python scripts/check_vendor_updates.py [--json] [--fail-on-stale]

Options
-------
--json          Output results as JSON (useful for CI integration).
--fail-on-stale Exit with code 1 if any submodule is behind upstream.
--no-network    Skip GitHub API call; compare against local remote refs only.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SUBMODULE_UPDATE_CMD = "git submodule update --remote {path}"

# Map submodule path → GitHub API URL for the tracking branch HEAD
SUBMODULE_UPSTREAM: dict[str, str] = {
    "vendor/ud-tools": "https://api.github.com/repos/D-sorganization/Tools/commits/main",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SubmoduleStatus:
    path: str
    pinned_sha: str
    upstream_sha: str
    is_current: bool
    message: str


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a git command and return stdout stripped.

    Raises
    ------
    subprocess.CalledProcessError
        If the command exits with non-zero status.
    """
    result = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def get_pinned_sha(submodule_path: str) -> str:
    """Return the commit SHA currently pinned for the submodule.

    Uses ``git ls-files --stage`` which works even when the submodule is not
    initialised (no clone required).
    """
    output = _run(["git", "ls-files", "--stage", submodule_path])
    # Format: <mode> <sha> <stage>   <path>
    for line in output.splitlines():
        if submodule_path in line:
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return "<unknown>"


def get_local_remote_sha(submodule_path: str) -> str:
    """Return the SHA of the upstream tracking branch using local remote refs.

    Does not require internet access; only uses already-fetched remote data.
    """
    try:
        # Get the tracking branch
        tracking = _run(
            ["git", "config", f"submodule.{submodule_path}.branch"],
        )
    except subprocess.CalledProcessError:
        tracking = "main"

    try:
        sha = _run(
            ["git", "rev-parse", f"origin/{tracking}"],
            cwd=REPO_ROOT / submodule_path,
        )
        return sha
    except subprocess.CalledProcessError:
        return "<unavailable>"


def get_github_sha(api_url: str) -> str:
    """Fetch the latest commit SHA from the GitHub REST API.

    Falls back to ``<api-error>`` on failure so the script stays non-crashing.
    """
    try:
        import urllib.request  # noqa: PLC0415

        req = urllib.request.Request(
            api_url, headers={"Accept": "application/vnd.github.v3+json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
            return str(data.get("sha", "<no-sha>"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("GitHub API call failed for %s: %s", api_url, exc)
        return "<api-error>"


# ---------------------------------------------------------------------------
# Core check logic
# ---------------------------------------------------------------------------


def check_submodule(
    path: str, api_url: str | None, use_network: bool
) -> SubmoduleStatus:
    """Check one submodule for staleness.

    Parameters
    ----------
    path:
        Relative path to the submodule inside the repository.
    api_url:
        GitHub API URL for the upstream branch HEAD (may be ``None``).
    use_network:
        If ``True`` and ``api_url`` is set, query GitHub; otherwise use
        local remote refs.

    Returns
    -------
    SubmoduleStatus
        Structured status report for this submodule.
    """
    pinned = get_pinned_sha(path)

    if use_network and api_url:
        upstream = get_github_sha(api_url)
    else:
        upstream = get_local_remote_sha(path)

    is_current = (
        pinned != "<unknown>"
        and upstream
        not in (
            "<unavailable>",
            "<api-error>",
            "<unknown>",
        )
        and pinned.startswith(upstream[:10])
        or pinned == upstream
    )

    if is_current:
        msg = f"✅  {path} is current ({pinned[:12]})"
    elif upstream in ("<unavailable>", "<api-error>"):
        msg = f"⚠️  {path}: could not determine upstream SHA (pinned={pinned[:12]})"
        is_current = True  # treat as non-blocking when we can't compare
    else:
        msg = (
            f"🔴  {path} is STALE\n"
            f"    Pinned:   {pinned[:12]}\n"
            f"    Upstream: {upstream[:12]}\n"
            f"    Update:   {SUBMODULE_UPDATE_CMD.format(path=path)}"
        )

    return SubmoduleStatus(
        path=path,
        pinned_sha=pinned,
        upstream_sha=upstream,
        is_current=is_current,
        message=msg,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check vendor submodule freshness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="Exit with code 1 if any submodule is behind upstream.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip GitHub API calls; compare against local remote refs only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code."""
    args = build_arg_parser().parse_args(argv)
    use_network = not args.no_network

    results: list[SubmoduleStatus] = []
    for path, api_url in SUBMODULE_UPSTREAM.items():
        status = check_submodule(path, api_url, use_network)
        results.append(status)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print("\n=== Vendor Submodule Freshness Report ===\n")
        for r in results:
            print(r.message)
        print()

    stale_count = sum(1 for r in results if not r.is_current)

    if stale_count > 0 and not args.json:
        print(
            f"⚠️  {stale_count} submodule(s) are behind upstream.\n"
            "   Run the following to update:\n"
            "   git submodule update --remote vendor/ud-tools\n"
            "   git add vendor/ud-tools\n"
            '   git commit -m "chore: sync vendor/ud-tools to latest Tools main"\n'
        )

    if args.fail_on_stale and stale_count > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
