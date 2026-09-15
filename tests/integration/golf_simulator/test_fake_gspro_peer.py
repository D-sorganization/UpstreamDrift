"""Integration tests for GSPro durable transport against a simulated TCP peer (GS-03, #10192).

Verifies:
- Clean TCP connection handshake and framed request/response round-trip.
- Reassembly of fragmented TCP stream frames without data loss.
- In-flight server drops transition delivery intent to AMBIGUOUS in the journal
  and enforce the no-auto-resend safety invariant.
- Buffer overflow enforcement (>64 KiB) disconnects cleanly.
- Error codes (501) map to REJECTED delivery state in the journal.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
import pytest

from src.shared.python.golf_simulator.adapters.gspro.codec import ResponseCategory
from src.shared.python.golf_simulator.adapters.gspro.transport import (
    FramingBufferOverflowError,
    GSProTransport,
    TransportError,
)
from src.shared.python.golf_simulator.journal import DeliveryStatus, ShotJournal

pytestmark = pytest.mark.integration


class FakeGSProPeer:
    """Async TCP fake server simulating GSPro Open Connect v1 endpoints."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port: int = 0
        self.server: asyncio.Server | None = None
        self.received_payloads: list[dict[str, Any]] = []
        self.behavior: str = (
            "normal"  # "normal", "fragmented", "drop", "overflow", "error_501"
        )

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, 0)
        sockets = self.server.sockets
        assert sockets is not None and len(sockets) > 0
        self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                # Parse received JSON payload
                try:
                    payload = json.loads(data.decode("utf-8"))
                    self.received_payloads.append(payload)
                except json.JSONDecodeError:
                    pass

                if self.behavior == "normal":
                    response = json.dumps(
                        {"Code": 200, "Message": "Shot received"}
                    ).encode("utf-8")
                    writer.write(response)
                    await writer.drain()
                elif self.behavior == "fragmented":
                    response = json.dumps(
                        {"Code": 200, "Message": "Shot received"}
                    ).encode("utf-8")
                    # Send in tiny 2-byte chunks
                    for i in range(0, len(response), 2):
                        writer.write(response[i : i + 2])
                        await writer.drain()
                        await asyncio.sleep(0.01)
                elif self.behavior == "drop":
                    # Abruptly close socket without responding
                    writer.close()
                    await writer.wait_closed()
                    return
                elif self.behavior == "overflow":
                    # Send huge chunk > 64 KiB
                    overflow_data = b"{" + (b"A" * 70000)
                    writer.write(overflow_data)
                    await writer.drain()
                elif self.behavior == "error_501":
                    response = json.dumps(
                        {"Code": 501, "Message": "Feature not supported"}
                    ).encode("utf-8")
                    writer.write(response)
                    await writer.drain()
        except (asyncio.CancelledError, ConnectionError, OSError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass


@pytest.mark.asyncio
async def test_successful_shot_round_trip() -> None:
    server = FakeGSProPeer()
    await server.start()
    journal = ShotJournal()
    transport = GSProTransport(host=server.host, port=server.port, journal=journal)

    try:
        await transport.connect()
        shot_payload = json.dumps(
            {"DeviceID": "UpstreamDrift", "ShotNumber": 42}
        ).encode("utf-8")
        receipt = await transport.send_shot("shot-42", shot_payload)

        assert receipt.code == 200
        assert receipt.category == ResponseCategory.CONFIRMED_ACCEPTED
        assert len(server.received_payloads) == 1
        assert server.received_payloads[0]["ShotNumber"] == 42

        # Verify journal status transitioned to ACKNOWLEDGED
        entry = journal.get_entry("shot-42")
        assert entry.status == DeliveryStatus.ACKNOWLEDGED
        assert entry.response_code == 200
    finally:
        await transport.disconnect()
        await server.stop()


@pytest.mark.asyncio
async def test_fragmented_stream_framer() -> None:
    server = FakeGSProPeer()
    server.behavior = "fragmented"
    await server.start()
    journal = ShotJournal()
    transport = GSProTransport(host=server.host, port=server.port, journal=journal)

    try:
        await transport.connect()
        shot_payload = json.dumps(
            {"DeviceID": "UpstreamDrift", "ShotNumber": 43}
        ).encode("utf-8")
        receipt = await transport.send_shot("shot-43", shot_payload)

        assert receipt.code == 200
        assert receipt.category == ResponseCategory.CONFIRMED_ACCEPTED
        entry = journal.get_entry("shot-43")
        assert entry.status == DeliveryStatus.ACKNOWLEDGED
    finally:
        await transport.disconnect()
        await server.stop()


@pytest.mark.asyncio
async def test_ambiguous_disconnect_handling() -> None:
    server = FakeGSProPeer()
    server.behavior = "drop"
    await server.start()
    journal = ShotJournal()
    transport = GSProTransport(host=server.host, port=server.port, journal=journal)

    try:
        await transport.connect()
        shot_payload = json.dumps(
            {"DeviceID": "UpstreamDrift", "ShotNumber": 44}
        ).encode("utf-8")

        with pytest.raises(TransportError, match="Connection closed"):
            await transport.send_shot("shot-44", shot_payload)

        # Invariant: Disconnect in-flight marks shot as AMBIGUOUS in the journal
        entry = journal.get_entry("shot-44")
        assert entry.status == DeliveryStatus.AMBIGUOUS
        assert "Connection closed" in (entry.error or "")

        # Verify no automatic retry occurred
        assert len(server.received_payloads) == 1
    finally:
        await transport.disconnect()
        await server.stop()


@pytest.mark.asyncio
async def test_buffer_overflow_protection() -> None:
    server = FakeGSProPeer()
    server.behavior = "overflow"
    await server.start()
    journal = ShotJournal()
    transport = GSProTransport(
        host=server.host, port=server.port, max_buffer_bytes=65536, journal=journal
    )

    try:
        await transport.connect()
        shot_payload = json.dumps(
            {"DeviceID": "UpstreamDrift", "ShotNumber": 45}
        ).encode("utf-8")

        with pytest.raises(FramingBufferOverflowError):
            await transport.send_shot("shot-45", shot_payload)
    finally:
        await transport.disconnect()
        await server.stop()


@pytest.mark.asyncio
async def test_rejection_mapping() -> None:
    server = FakeGSProPeer()
    server.behavior = "error_501"
    await server.start()
    journal = ShotJournal()
    transport = GSProTransport(host=server.host, port=server.port, journal=journal)

    try:
        await transport.connect()
        shot_payload = json.dumps(
            {"DeviceID": "UpstreamDrift", "ShotNumber": 46}
        ).encode("utf-8")
        receipt = await transport.send_shot("shot-46", shot_payload)

        assert receipt.code == 501
        assert receipt.category == ResponseCategory.ERROR_REJECTED

        entry = journal.get_entry("shot-46")
        assert entry.status == DeliveryStatus.REJECTED
        assert entry.response_code == 501
    finally:
        await transport.disconnect()
        await server.stop()
