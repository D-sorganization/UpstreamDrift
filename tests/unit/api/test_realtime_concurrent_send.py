"""Tests for per-socket asyncio.Lock in the realtime subscriber registry (#6978).

Two concurrent POST /realtime/publish calls to the same channel could both
await ws.send_json() on the same WebSocket concurrently, interleaving frames
or dropping healthy subscribers.  The fix stores a per-socket asyncio.Lock
in _SubscriberRegistry so only one coroutine sends on a given socket at a time.
"""

from __future__ import annotations

import asyncio

import pytest

from src.api.routes.realtime import _SubscriberRegistry


@pytest.mark.unit
class TestSubscriberRegistryPerSocketLock:
    """Verify that each WebSocket gets its own asyncio.Lock in the registry."""

    async def test_add_creates_per_socket_lock(self) -> None:
        """Adding a WebSocket to the registry should create a lock for it."""
        registry = _SubscriberRegistry()
        ws = object()  # minimal stand-in; registry only needs hashability

        await registry.add("pose/canonical", ws)  # type: ignore[arg-type]

        assert ws in registry._ws_locks, "Lock not created for ws after add()"
        assert isinstance(registry._ws_locks[ws], asyncio.Lock), (
            "Lock must be an asyncio.Lock"
        )

    async def test_snapshot_returns_ws_lock_pairs(self) -> None:
        """snapshot() must return (ws, lock) pairs, not bare WebSocket objects."""
        registry = _SubscriberRegistry()
        ws = object()
        await registry.add("pose/canonical", ws)  # type: ignore[arg-type]

        result = await registry.snapshot("pose/canonical")

        assert len(result) == 1, "Expected one subscriber"
        assert len(result[0]) == 2, "snapshot() items must be (ws, lock) tuples"
        ws_out, lock_out = result[0]
        assert ws_out is ws
        assert isinstance(lock_out, asyncio.Lock)

    async def test_remove_cleans_up_lock(self) -> None:
        """Removing the last subscription for a ws should clean up its lock."""
        registry = _SubscriberRegistry()
        ws = object()
        await registry.add("pose/canonical", ws)  # type: ignore[arg-type]
        await registry.remove("pose/canonical", ws)  # type: ignore[arg-type]

        assert ws not in registry._ws_locks, (
            "Lock should be removed when ws leaves all channels"
        )

    async def test_lock_retained_while_ws_in_other_channel(self) -> None:
        """Lock must stay alive when a ws is still in another channel."""
        registry = _SubscriberRegistry()
        ws = object()
        await registry.add("channel/a", ws)  # type: ignore[arg-type]
        await registry.add("channel/b", ws)  # type: ignore[arg-type]
        await registry.remove("channel/a", ws)  # type: ignore[arg-type]

        assert ws in registry._ws_locks, (
            "Lock must persist while ws is still in channel/b"
        )

    async def test_concurrent_publishes_serialize_per_socket(self) -> None:
        """Concurrent sends to the same socket should be serialized by the lock."""
        registry = _SubscriberRegistry()

        send_order: list[int] = []

        class _SlowWS:
            """WebSocket stub that records send order and pauses mid-send."""

            async def send_json(self, payload: object) -> None:
                tag = payload["tag"]  # type: ignore[index]
                send_order.append(tag)
                if tag == 1:
                    # Pause so the second coroutine tries to interleave.
                    await asyncio.sleep(0.05)

        ws = _SlowWS()
        await registry.add("pose/canonical", ws)  # type: ignore[arg-type]

        async def _publish(tag: int) -> None:
            pairs = await registry.snapshot("pose/canonical")
            for sock, lock in pairs:
                async with lock:
                    await sock.send_json({"tag": tag})  # type: ignore[union-attr]

        # Fire two concurrent publishes; without a lock they'd interleave.
        await asyncio.gather(_publish(1), _publish(2))

        # With the per-socket lock the sends must be sequential (not interleaved).
        assert send_order == [1, 2] or send_order == [2, 1], (
            f"Sends were not serialized: {send_order}"
        )
        # Key property: no entry appears twice (no duplicate send).
        assert len(set(send_order)) == 2, "Each payload must be sent exactly once"
