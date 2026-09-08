"""Hardware-free contracts for USB topology facts."""

from __future__ import annotations

import pytest

from src.motion_capture.rig.topology import (
    MAX_CASCADED_HUBS,
    PERIODIC_BUDGET_BYTES,
    RESERVE_BYTES,
    CameraLocation,
    attach_capture_indices,
    cameras_per_bus,
    conflicting_identities,
    derive_camera,
    parse_dshow_listing,
    path_identity,
    predict_streaming,
)

pytestmark = pytest.mark.unit

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


def _chain(*rows: tuple[str, str, str]) -> list[dict[str, str]]:
    return [{"id": i, "name": n, "loc": loc} for i, n, loc in rows]


def _cam(
    identity: str, port: int, hub: str = "USB\\ROOT_HUB30\\9&1d291187&0&0"
) -> CameraLocation:
    return CameraLocation(
        camera=f"USB\\VID_32E4&PID_5234&MI_00\\{identity}",
        composite=None,
        serial=None,
        identity=identity,
        root_hub=hub,
        root_port=port,
        host="PCI\\VEN_8086&DEV_15C1",
        hub_depth=0,
    )


def test_budget_constants_encode_the_measured_limit() -> None:
    assert PERIODIC_BUDGET_BYTES == 6000
    assert RESERVE_BYTES == 3060
    assert cameras_per_bus() == 1
    assert MAX_CASCADED_HUBS == 5


def test_parse_dshow_listing_keeps_video_order_and_skips_audio() -> None:
    order = parse_dshow_listing(DSHOW_LISTING)
    assert [name for name, _ in order] == [
        "HP Wide Vision 5MP Camera",
        "Global Shutter Camera",
        "Global Shutter Camera",
    ]
    assert order[1][1] == "USB\\VID_32E4&PID_5234&MI_00\\6&FADBF3B&0&0000"


def test_derive_camera_reads_serial_root_port_and_hub_depth() -> None:
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
    cam = derive_camera(entry)
    assert cam.serial == "2605160001"
    assert cam.identity == "2605160001"
    assert cam.root_hub == "USB\\ROOT_HUB30\\4&2f6b3c63&0&0"
    assert cam.root_port == 4
    assert cam.host is not None and cam.host.startswith("PCI\\VEN_8086&DEV_51ED")
    assert cam.hub_depth == 2
    assert cam.bus_key == (cam.host, cam.root_hub, 4)
    assert cam.to_dict()["identity"] == "2605160001"


def test_derive_camera_falls_back_to_port_path_without_serial() -> None:
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
    cam = derive_camera(entry)
    assert cam.serial is None
    assert cam.identity == path_identity(entry["camera"]) == "path_9-78FC85E-0-0000"
    assert cam.root_port == 1
    assert cam.hub_depth == 0


def test_derive_camera_rejects_malformed_entry() -> None:
    with pytest.raises(Exception, match="camera and chain"):
        derive_camera({"camera": "x"})


def test_predict_allows_exactly_one_camera_per_root_port() -> None:
    shared, lines = predict_streaming([_cam("a", 4), _cam("b", 4), _cam("c", 4)])
    assert shared == 1
    assert any("CONFLICT" in line for line in lines)
    assert set(conflicting_identities([_cam("a", 4), _cam("b", 4), _cam("c", 4)])) == {
        "a",
        "b",
        "c",
    }

    distinct, lines = predict_streaming([_cam("a", 4), _cam("b", 5), _cam("c", 6)])
    assert distinct == 3
    assert not any("CONFLICT" in line for line in lines)
    assert conflicting_identities([_cam("a", 4), _cam("b", 5)]) == []


def test_attach_capture_indices_uses_dshow_order_and_sorts() -> None:
    cams = [
        CameraLocation(
            camera="USB\\VID_32E4&PID_5234&MI_00\\9&2A7EE39F&0&0000",
            composite=None,
            serial="2601240001",
            identity="2601240001",
            root_hub=None,
            root_port=None,
            host=None,
            hub_depth=0,
        ),
        CameraLocation(
            camera="USB\\VID_32E4&PID_5234&MI_00\\6&FADBF3B&0&0000",
            composite=None,
            serial="2605160001",
            identity="2605160001",
            root_hub=None,
            root_port=None,
            host=None,
            hub_depth=0,
        ),
    ]
    ordered = attach_capture_indices(cams, parse_dshow_listing(DSHOW_LISTING))
    assert [c.index for c in ordered] == [1, 2]
    assert ordered[0].identity == "2605160001"
