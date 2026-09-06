"""Hardware-free contracts for the USB camera rig diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "diagnose_mocap_camera_rig.py"


@pytest.fixture(scope="module")
def rig():
    spec = importlib.util.spec_from_file_location("diagnose_mocap_camera_rig", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve postponed annotations through sys.modules[__module__]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DSHOW_LISTING = """\
[dshow @ 0000018f] "HP Wide Vision 5MP Camera" (video)
[dshow @ 0000018f]   Alternative name "@device_pnp_\\\\?\\usb#vid_30c9&pid_00ab&mi_00#6&10f595f8&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
[dshow @ 0000018f] "Global Shutter Camera" (video)
[dshow @ 0000018f]   Alternative name "@device_pnp_\\\\?\\usb#vid_32e4&pid_5234&mi_00#6&fadbf3b&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
[dshow @ 0000018f] "Global Shutter Camera" (video)
[dshow @ 0000018f]   Alternative name "@device_pnp_\\\\?\\usb#vid_32e4&pid_5234&mi_00#9&2a7ee39f&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global"
[dshow @ 0000018f] "Microphone (Realtek)" (audio)
[dshow @ 0000018f]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{73E78BC9}"
"""


def _chain(*ids_and_names: tuple[str, str, str]) -> list[dict[str, str]]:
    return [{"id": i, "name": n, "loc": loc} for i, n, loc in ids_and_names]


def test_parse_dshow_listing_keeps_video_order_and_skips_audio(rig) -> None:
    order = rig.parse_dshow_listing(DSHOW_LISTING)
    assert [name for name, _ in order] == [
        "HP Wide Vision 5MP Camera",
        "Global Shutter Camera",
        "Global Shutter Camera",
    ]
    assert order[1][1] == "USB\\VID_32E4&PID_5234&MI_00\\6&FADBF3B&0&0000"


def test_derive_camera_reads_serial_root_port_and_hub_depth(rig) -> None:
    entry = {
        "camera": "USB\\VID_32E4&PID_5234&MI_00\\6&FADBF3B&0&0000",
        "chain": _chain(
            (
                "USB\\VID_32E4&PID_5234&MI_00\\6&FADBF3B&0&0000",
                "Global Shutter Camera",
                "",
            ),
            (
                "USB\\VID_32E4&PID_5234\\2605160001",
                "USB Composite Device",
                "Port_#0004.Hub_#0021",
            ),
            (
                "USB\\VID_0BDA&PID_5411\\9&303bf606&0&4",
                "Generic USB Hub",
                "Port_#0004.Hub_#0019",
            ),
            (
                "USB\\VID_2188&PID_5802\\5&285eeee0&0&4",
                "Generic USB Hub",
                "Port_#0004.Hub_#0002",
            ),
            ("USB\\ROOT_HUB30\\4&2f6b3c63&0&0", "USB Root Hub (USB 3.0)", ""),
            (
                "PCI\\VEN_8086&DEV_51ED\\3&11583659&0&A0",
                "Intel(R) USB 3.10 eXtensible Host Controller",
                "",
            ),
        ),
    }
    cam = rig.derive_camera(entry)
    assert cam.serial == "2605160001"
    assert cam.identity == "2605160001"
    assert cam.root_hub == "USB\\ROOT_HUB30\\4&2f6b3c63&0&0"
    assert cam.root_port == 4
    assert cam.host.startswith("PCI\\VEN_8086&DEV_51ED")
    assert cam.hub_depth == 2


def test_derive_camera_falls_back_to_port_path_without_serial(rig) -> None:
    entry = {
        "camera": "USB\\VID_32E4&PID_5234&MI_00\\9&78FC85E&0&0000",
        "chain": _chain(
            (
                "USB\\VID_32E4&PID_5234&MI_00\\9&78FC85E&0&0000",
                "Global Shutter Camera",
                "",
            ),
            (
                "USB\\VID_32E4&PID_5234\\8&2956EDF6&0&1",
                "USB Composite Device",
                "Port_#0001.Hub_#0007",
            ),
            ("USB\\ROOT_HUB30\\9&1d291187&0&0", "USB Root Hub (USB 3.0)", ""),
        ),
    }
    cam = rig.derive_camera(entry)
    assert cam.serial is None
    assert cam.identity == "path_9-78FC85E-0-0000"
    assert cam.root_port == 1
    assert cam.hub_depth == 0


def test_predict_allows_exactly_one_camera_per_root_port(rig) -> None:
    def cam(port: int) -> object:
        return rig.Camera(
            camera=f"cam{port}",
            composite=None,
            serial=None,
            identity=f"cam{port}",
            root_hub="USB\\ROOT_HUB30\\9&1d291187&0&0",
            root_port=port,
            host="PCI\\VEN_8086&DEV_15C1",
            hub_depth=0,
        )

    shared, lines = rig.predict([cam(4), cam(4), cam(4)])
    assert shared == 1
    assert any("CONFLICT" in line for line in lines)

    distinct, lines = rig.predict([cam(4), cam(5), cam(6)])
    assert distinct == 3
    assert not any("CONFLICT" in line for line in lines)
