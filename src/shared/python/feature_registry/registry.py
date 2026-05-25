"""Runtime feature registry.

Single source of truth for "is this feature available *right now* in
this Python interpreter, given its installed packages and the bundled
asset directories?"

The registry aggregates per-feature probe outcomes
(:mod:`src.shared.python.feature_registry.probes`) keyed by the
feature definitions in :mod:`src.shared.python.feature_registry.features`.

Consumers
---------
* CLI: ``python -m src.shared.python.feature_registry`` prints a
  table; ``python -m src.shared.python.feature_registry --json``
  prints JSON.
* REST API: ``src.api.routes.capabilities`` returns the same JSON
  payload over HTTP.
* Launcher: the missing-dependency dialog looks up a single feature.
* CI: smoke tests assert that the expected feature set is available
  in each Docker profile.

Design by Contract
------------------
Invariants:
    * Every entry in :data:`src.shared.python.feature_registry.features.FEATURES`
      that declares a ``probe_key`` MUST have a matching entry in
      :data:`src.shared.python.feature_registry.probes.PROBES`.
      This is verified at import time by :func:`_validate_probe_mapping`.

Postconditions:
    * :meth:`CapabilityRegistry.snapshot` returns one
      :class:`FeatureReport` per registered feature, in registration
      order.
    * :meth:`CapabilityRegistry.refresh` invalidates Python's import
      caches; subsequent probes see freshly-installed packages without
      requiring a process restart.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from src.shared.python.feature_registry.features import (
    FEATURES,
    Feature,
    get_feature,
)
from src.shared.python.feature_registry.probes import PROBES, ProbeOutcome

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureReport:
    """Public report describing one feature's runtime availability.

    Returned by :meth:`CapabilityRegistry.check` and aggregated by
    :meth:`CapabilityRegistry.snapshot`.
    """

    name: str
    display_name: str
    available: bool
    version: str | None
    tier: str
    docker_stage: str | None
    install_channel: str
    install_command: str
    pip_extra: str | None
    approx_size_mb: int
    message: str
    missing: tuple[str, ...]
    depends_on: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict representation.

        Includes a ``status`` field (``"AVAILABLE"`` or ``"UNAVAILABLE"``)
        in addition to the boolean ``available`` flag, so shell consumers
        (e.g. ``docker-smoke.yml``) can grep for an unambiguous string.
        """
        data = asdict(self)
        data["missing"] = list(self.missing)
        data["depends_on"] = list(self.depends_on)
        data["status"] = "AVAILABLE" if self.available else "UNAVAILABLE"
        return data


def _suite_root() -> Path:
    """Return the repository root directory.

    Resolved relative to this file: ``src/shared/python/feature_registry/registry.py``
    is four levels deep, so ``parents[4]`` is the repo root.
    """
    return Path(__file__).resolve().parents[4]


def _validate_probe_mapping() -> None:
    """DbC: ensure every feature with a ``probe_key`` has a probe.

    Raised at import time as a guardrail. If you add a feature with a
    probe_key, this enforces that you also register the probe.
    """
    missing = [
        f.name
        for f in FEATURES
        if f.probe_key is not None and f.probe_key not in PROBES
    ]
    if missing:
        raise RuntimeError(
            "feature_registry invariant violated: features "
            f"{missing} declare a probe_key but no probe is registered. "
            "Add an entry to src/shared/python/feature_registry/probes.py::PROBES."
        )


_validate_probe_mapping()


class CapabilityRegistry:
    """Thread-safe aggregator over per-feature probes.

    The registry caches the latest probe outcome per feature. Probes
    can be expensive (some import native libraries), so callers should
    treat :meth:`snapshot` as the read path; :meth:`refresh` is the
    write path that re-runs probes.
    """

    def __init__(self, suite_root: Path | None = None) -> None:
        """Initialize the registry.

        Args:
            suite_root: Override for the repository root (used by tests).
                Defaults to the auto-detected root.
        """
        self._suite_root = suite_root if suite_root is not None else _suite_root()
        self._lock = Lock()
        self._cache: dict[str, FeatureReport] = {}
        self._populated = False

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def snapshot(self) -> tuple[FeatureReport, ...]:
        """Return one :class:`FeatureReport` per registered feature.

        On the first call, every probe is executed; subsequent calls
        return the cached values until :meth:`refresh` is invoked.
        """
        if not self._populated:
            self.refresh()
        with self._lock:
            return tuple(self._cache[f.name] for f in FEATURES)

    def check(self, name: str) -> FeatureReport:
        """Return the report for a single feature.

        Triggers a full population on first use, then reads from the
        cache. Use :meth:`refresh_one` to re-run a single probe.

        Raises:
            KeyError: if ``name`` is not a registered feature.
        """
        if not self._populated:
            self.refresh()
        with self._lock:
            try:
                return self._cache[get_feature(name).name]
            except KeyError as exc:
                raise KeyError(f"Unknown feature {name!r}") from exc

    def is_available(self, name: str) -> bool:
        """Convenience helper — ``True`` if the feature is usable."""
        return self.check(name).available

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def refresh(self) -> tuple[FeatureReport, ...]:
        """Re-run every probe and replace the cache.

        Call this after an in-process ``pip install`` so the registry
        reflects the new state. Invalidates Python's import-finder
        caches so freshly installed packages are visible.
        """
        importlib.invalidate_caches()
        new_cache: dict[str, FeatureReport] = {}
        for feature in FEATURES:
            new_cache[feature.name] = self._evaluate(feature)
        with self._lock:
            self._cache = new_cache
            self._populated = True
        return tuple(new_cache[f.name] for f in FEATURES)

    def refresh_one(self, name: str) -> FeatureReport:
        """Re-run a single probe and update the cache for that feature."""
        feature = get_feature(name)
        importlib.invalidate_caches()
        report = self._evaluate(feature)
        with self._lock:
            self._cache[feature.name] = report
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate(self, feature: Feature) -> FeatureReport:
        """Run the probe (if any) and build a :class:`FeatureReport`."""
        if feature.probe_key is None:
            outcome = ProbeOutcome(
                available=True,
                version=None,
                message=f"{feature.display_name} (no probe — always available)",
            )
        else:
            probe = PROBES.get(feature.probe_key)
            if probe is None:
                # Defensive — _validate_probe_mapping should have caught this
                outcome = ProbeOutcome(
                    available=False,
                    version=None,
                    message=(
                        f"No probe registered for {feature.probe_key!r}; "
                        "registry misconfigured."
                    ),
                    missing=(feature.probe_key,),
                )
            else:
                try:
                    outcome = probe(self._suite_root)
                except (ImportError, OSError, RuntimeError) as exc:
                    # A probe should not raise — but if one does we
                    # surface it as UNAVAILABLE rather than crashing.
                    logger.warning(
                        "Probe for %s raised %s; treating as unavailable",
                        feature.name,
                        exc,
                    )
                    outcome = ProbeOutcome(
                        available=False,
                        version=None,
                        message=f"Probe error: {exc}",
                        missing=(feature.probe_key,),
                    )
        return FeatureReport(
            name=feature.name,
            display_name=feature.display_name,
            available=outcome.available,
            version=outcome.version,
            tier=feature.tier,
            docker_stage=feature.docker_stage,
            install_channel=feature.install_channel,
            install_command=feature.install_command,
            pip_extra=feature.pip_extra,
            approx_size_mb=feature.approx_size_mb,
            message=outcome.message,
            missing=outcome.missing,
            depends_on=feature.depends_on,
        )


# ---------------------------------------------------------------------------
# Singleton accessor — preserves cache across callers in the same process.
# ---------------------------------------------------------------------------

_REGISTRY: CapabilityRegistry | None = None
_REGISTRY_LOCK = Lock()


def get_registry() -> CapabilityRegistry:
    """Return the process-wide :class:`CapabilityRegistry` singleton."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = CapabilityRegistry()
    return _REGISTRY


def refresh() -> tuple[FeatureReport, ...]:
    """Convenience: refresh the singleton registry."""
    return get_registry().refresh()


# ---------------------------------------------------------------------------
# CLI: python -m src.shared.python.feature_registry [--json|--check NAME]
# ---------------------------------------------------------------------------


def _format_table(reports: tuple[FeatureReport, ...]) -> str:
    """Render reports as a fixed-width table for stdout."""
    rows = [
        ("FEATURE", "STATUS", "VERSION", "TIER", "SIZE(MB)", "NOTE"),
    ]
    for r in reports:
        rows.append(
            (
                r.name,
                "ok" if r.available else "missing",
                r.version or "-",
                r.tier,
                str(r.approx_size_mb),
                # Truncate diagnostic so the table stays one line per feature.
                (r.message[:60] + "…") if len(r.message) > 60 else r.message,
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("-" * len(line))
    return "\n".join(lines)


def _write(stream, text: str) -> None:
    stream.write(text)
    if not text.endswith("\n"):
        stream.write("\n")


def _main(argv: list[str]) -> int:
    """CLI entry point — used by ``python -m`` and ``upstream-drift caps``.

    Writes to ``sys.stdout`` / ``sys.stderr`` directly rather than via
    ``print`` so the registry module complies with the repo's
    ``no print() in src/`` rule (CLAUDE.md CI requirements).
    """
    as_json = "--json" in argv
    check_idx = argv.index("--check") if "--check" in argv else -1

    registry = get_registry()
    out = sys.stdout
    err = sys.stderr

    if check_idx >= 0:
        if check_idx + 1 >= len(argv):
            _write(err, "error: --check requires a feature name")
            return 2
        name = argv[check_idx + 1]
        try:
            report = registry.check(name)
        except KeyError as exc:
            _write(err, f"error: {exc}")
            return 2
        if as_json:
            _write(out, json.dumps(report.to_dict(), indent=2))
        else:
            status_label = "AVAILABLE" if report.available else "UNAVAILABLE"
            _write(out, f"{report.name}: {status_label}")
            _write(out, f"  version: {report.version or '-'}")
            _write(out, f"  message: {report.message}")
            if not report.available:
                _write(out, f"  fix:     {report.install_command}")
        return 0 if report.available else 1

    reports = registry.snapshot()
    if as_json:
        _write(out, json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        _write(out, _format_table(reports))
    return 0
