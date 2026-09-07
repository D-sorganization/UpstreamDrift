"""USB topology facts for the camera rig.

Measured on 2026-09-06 (``docs/motion_capture/usb_camera_rig_bringup.md``):
the ELP AR0234 "Global Shutter Camera" requests its largest isochronous
alt-setting — 3 transactions x 1020 bytes = 3060 bytes per microframe — for
every video format, and USB 2.0 High-Speed allows 6000 bytes of periodic
traffic per microframe. Two of these cameras therefore never fit on one USB
2.0 bus, and xHCI accounts that budget per root port. The pure functions here
turn Windows PnP hub chains into :class:`CameraLocation` records and predict
how many cameras can stream; the Windows probes at the bottom gather the raw
data and are the only parts that touch the host.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

VENDOR_ID = "32E4"
PERIODIC_BUDGET_BYTES = 6000  # 80 % of a 7500-byte USB 2.0 High-Speed microframe
RESERVE_BYTES = 3060  # alt-setting 0x0B: 3 transactions x 1020 bytes per microframe
MAX_CASCADED_HUBS = 5

_DSHOW_NAME = re.compile(r'\]\s+"(.+?)"\s+\(video\)')
_DSHOW_ALT = re.compile(r'Alternative name\s+"@device_pnp_\\\\\?\\(.+?)#\{')
_ROOT_PORT = re.compile(r"Port_#(\d+)")

_PS_TOPOLOGY = r"""
$ErrorActionPreference = 'SilentlyContinue'
$cams = Get-PnpDevice -PresentOnly -Class Camera |
  Where-Object { $_.InstanceId -match 'VID_@VENDOR@' }
$out = @()
foreach ($c in $cams) {
  $chain = @(); $cur = $c.InstanceId
  while ($cur) {
    $d = Get-PnpDevice -InstanceId $cur
    if (-not $d) { break }
    $loc = (Get-PnpDeviceProperty -InstanceId $cur `
      -KeyName 'DEVPKEY_Device_LocationInfo').Data
    $chain += [PSCustomObject]@{ name = $d.FriendlyName; id = $cur; loc = [string]$loc }
    if ($cur -match '^PCI\\') { break }
    $cur = (Get-PnpDeviceProperty -InstanceId $cur -KeyName 'DEVPKEY_Device_Parent').Data
  }
  $out += [PSCustomObject]@{ camera = $c.InstanceId; chain = $chain }
}
ConvertTo-Json -InputObject @($out) -Depth 5 -Compress
"""


@dataclass
class CameraLocation:
    """One enumerated camera and its position in the USB tree.

    ``identity`` is the USB serial when the unit exposes one, otherwise a
    stable ``path_<instance tail>`` derived from the camera's port path.
    """

    camera: str
    composite: str | None
    serial: str | None
    identity: str
    root_hub: str | None
    root_port: int | None
    host: str | None
    hub_depth: int
    chain: list[dict[str, str]] = field(default_factory=list)
    index: int | None = None

    @property
    def bus_key(self) -> tuple[str | None, str | None, int | None]:
        """The bandwidth domain this camera competes in."""
        return (self.host, self.root_hub, self.root_port)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def path_identity(camera_instance_id: str) -> str:
    """Stable identity for a unit without a serial, from its instance path."""
    require(bool(camera_instance_id), "camera_instance_id must be non-empty")
    return "path_" + camera_instance_id.split("\\")[-1].replace("&", "-")


def derive_camera(entry: dict[str, Any], vendor: str = VENDOR_ID) -> CameraLocation:
    """Build a :class:`CameraLocation` from one PnP chain (camera first, host last).

    Preconditions: ``entry`` has ``camera`` (instance id) and ``chain`` (list of
    ``{name, id, loc}`` dicts ordered from the camera up to the host controller).
    Postcondition: ``identity`` is non-empty; ``hub_depth`` counts non-root hubs.
    """
    require("camera" in entry and "chain" in entry, "entry needs camera and chain")
    raw = entry["chain"]
    chain: list[dict[str, str]] = raw if isinstance(raw, list) else [raw]
    composite_re = re.compile(
        rf"USB\\VID_{vendor}&PID_[0-9A-F]{{4}}\\[^\\]+$", re.IGNORECASE
    )
    composite = next((x for x in chain if composite_re.match(x["id"])), None)
    tail = composite["id"].rsplit("\\", 1)[1] if composite else ""
    serial = tail if tail and "&" not in tail else None
    root_idx = next(
        (i for i, x in enumerate(chain) if x["id"].upper().startswith("USB\\ROOT_HUB")),
        None,
    )
    below = chain[root_idx - 1] if root_idx else None
    port = _ROOT_PORT.search(below.get("loc", "")) if below else None
    return CameraLocation(
        camera=entry["camera"],
        composite=composite["id"] if composite else None,
        serial=serial,
        identity=serial or path_identity(entry["camera"]),
        root_hub=chain[root_idx]["id"] if root_idx is not None else None,
        root_port=int(port.group(1)) if port else None,
        host=next(
            (x["id"] for x in chain if x["id"].upper().startswith("PCI\\")), None
        ),
        hub_depth=sum(
            1 for x in chain if "Hub" in x["name"] and "Root" not in x["name"]
        ),
        chain=chain,
    )


def parse_dshow_listing(text: str) -> list[tuple[str, str]]:
    """Return ``(friendly name, instance id)`` per DirectShow video device, in order.

    OpenCV's MSMF and DSHOW backends index devices in this order. The
    alternative name ``usb#vid_...#6&fadbf3b&0&0000`` maps to the PnP instance
    id ``USB\\VID_...\\6&FADBF3B&0&0000``.
    """
    order: list[tuple[str, str]] = []
    name: str | None = None
    for line in text.splitlines():
        if match := _DSHOW_NAME.search(line):
            name = match.group(1)
            continue
        if name is not None and (match := _DSHOW_ALT.search(line)):
            order.append((name, match.group(1).replace("#", "\\").upper()))
            name = None
    return order


def cameras_per_bus() -> int:
    """How many cameras fit in one USB 2.0 periodic budget (currently 1)."""
    return PERIODIC_BUDGET_BYTES // RESERVE_BYTES


def predict_streaming(cams: list[CameraLocation]) -> tuple[int, list[str]]:
    """Predict how many cameras can stream and describe each bandwidth domain.

    Returns ``(expected_streaming, report_lines)``. A domain holding more
    cameras than :func:`cameras_per_bus` is flagged ``CONFLICT``.
    """
    groups: dict[tuple[str | None, str | None, int | None], list[CameraLocation]] = {}
    for cam in cams:
        groups.setdefault(cam.bus_key, []).append(cam)
    fit = cameras_per_bus()
    expected = 0
    lines = []
    for (_, hub, port), group in groups.items():
        expected += min(len(group), fit)
        flag = "   <-- CONFLICT" if len(group) > fit else ""
        lines.append(
            f"  root_port {port} of {hub}: {len(group)} camera(s) -> "
            f"{min(len(group), fit)} can stream{flag}"
        )
    return expected, lines


def conflicting_identities(cams: list[CameraLocation]) -> list[str]:
    """Identities that share a bandwidth domain beyond its capacity."""
    groups: dict[tuple[str | None, str | None, int | None], list[str]] = {}
    for cam in cams:
        groups.setdefault(cam.bus_key, []).append(cam.identity)
    fit = cameras_per_bus()
    return [ident for group in groups.values() if len(group) > fit for ident in group]


# ---------------------------------------------------------------------------
# Windows probes (the only host-touching code in this module)
# ---------------------------------------------------------------------------


def query_topology(
    vendor: str = VENDOR_ID, timeout_s: float = 600
) -> list[CameraLocation]:
    """Walk every camera's hub chain through Windows PnP (seconds per hub tier)."""
    require(timeout_s > 0, "timeout_s must be positive", timeout_s)
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            _PS_TOPOLOGY.replace("@VENDOR@", vendor),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    data = json.loads(result.stdout.strip() or "[]")
    entries = [data] if isinstance(data, dict) else data
    return [derive_camera(e, vendor) for e in entries]


def dshow_order(timeout_s: float = 60) -> list[tuple[str, str]]:
    """List DirectShow video devices with the bundled imageio-ffmpeg binary."""
    import imageio_ffmpeg

    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return parse_dshow_listing(result.stderr + result.stdout)


def attach_capture_indices(
    cams: list[CameraLocation], order: list[tuple[str, str]]
) -> list[CameraLocation]:
    """Attach OpenCV capture indices from DirectShow order; sort by index."""
    index_by_path = {path: i for i, (_, path) in enumerate(order)}
    for cam in cams:
        cam.index = index_by_path.get(cam.camera.upper())
    cams.sort(key=lambda c: (c.index is None, c.index or 0))
    return cams
