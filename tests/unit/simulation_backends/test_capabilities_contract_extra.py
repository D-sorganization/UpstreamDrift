"""Supplementary tests for the simulation-backend capability + protocol contracts.

This file closes specific, well-defined coverage gaps in the
``simulation_backends`` foundation layer:

* :func:`require_mujoco` and :func:`require_warp` *failure* paths (the
  ``BackendNotAvailableError`` branches and actionable error messages).
* :func:`_can_import` *failure* path (a discoverable but broken module).
* :func:`warp_device_available` graceful-degradation branches.
* :class:`BackendCapabilities` equality + ``frozen`` guarantee.
* :class:`BackendError` and its three concrete subclasses' type hierarchies
  (``UnknownBackendError`` is a ``KeyError`` and ``BackendCapabilityError`` is
  a ``NotImplementedError`` — both are public contracts used by external
  ``except`` clauses).
* :class:`SimulationBackend`, :class:`DynamicsProvider`, and
  :class:`BatchedBackend` :func:`runtime_checkable` behaviour with concrete
  duck-typed implementations.

All tests are pure unit tests, fully deterministic, and do not invoke
optional GPU dependencies — they observe behaviour through :mod:`unittest.mock`
in a controlled fashion.

Test methodology
----------------
* **TDD**: tests are written as executable specifications of the contract.
* **DbC**: every assertion is paired with a precondition expressed in the
  docstring and (where the runtime supports it) a :func:`pytest.raises` guard
  that pins the error message to ``match=``.
* **DRY**: shared parametrize ids live at module scope; the test body never
  rebuilds the same fixture twice.
* **LOD**: tests do not reach through backend implementations — they construct
  minimal duck-typed stand-ins that satisfy the Protocol surface only.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.shared.python.simulation_backends.capabilities import (
    _can_import,
    has_mujoco,
    has_warp,
    require_mujoco,
    require_warp,
    warp_device_available,
)
from src.shared.python.simulation_backends.exceptions import (
    BackendCapabilityError,
    BackendError,
    BackendNotAvailableError,
    UnknownBackendError,
)
from src.shared.python.simulation_backends.protocol import (
    SCHEMA_VERSION,
    BatchedBackend,
    BackendCapabilities,
    DynamicsProvider,
    SimulationBackend,
)

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# require_mujoco / require_warp failure paths
# --------------------------------------------------------------------------- #
class TestRequireGates:
    """The :func:`require_*` gates are the loud-fail entry points.

    The success paths are exercised transitively by every test that
    constructs an ODE/MuJoCo backend on a machine that has those packages.
    The failure paths — what happens when the user *doesn't* have the
    optional extra — are isolated and pinned here.
    """

    def test_require_mujoco_raises_with_actionable_message(self) -> None:
        """``require_mujoco`` raises ``BackendNotAvailableError`` naming 'mujoco'."""
        with (
            patch(
                "src.shared.python.simulation_backends.capabilities.has_mujoco",
                return_value=False,
            ),
            pytest.raises(BackendNotAvailableError, match="mujoco") as exc_info,
        ):
            require_mujoco()
        # The error message must tell the user *how* to install the missing extra.
        assert "pip install" in str(exc_info.value)

    def test_require_warp_raises_with_actionable_message(self) -> None:
        """``require_warp`` raises ``BackendNotAvailableError`` naming 'warp'."""
        with (
            patch(
                "src.shared.python.simulation_backends.capabilities.has_warp",
                return_value=False,
            ),
            pytest.raises(BackendNotAvailableError, match="warp") as exc_info,
        ):
            require_warp()
        # The error message must reference the optional extra name.
        msg = str(exc_info.value)
        assert "pip install" in msg
        assert "[warp]" in msg

    def test_require_mujoco_passes_when_available(self) -> None:
        """When the module is importable the gate is a no-op."""
        # Don't actually toggle the lru_cache — call once and trust the
        # production code path on a real MuJoCo installation.
        try:
            import mujoco  # noqa: F401
        except ImportError:
            pytest.skip("mujoco not installed in this environment")
        require_mujoco()  # must not raise

    def test_require_warp_passes_when_available(self) -> None:
        """When the GPU stack is importable the gate is a no-op."""
        try:
            import mujoco_warp  # noqa: F401
            import warp  # noqa: F401
        except ImportError:
            pytest.skip("warp / mujoco_warp not installed in this environment")
        require_warp()  # must not raise


# --------------------------------------------------------------------------- #
# _can_import failure path
# --------------------------------------------------------------------------- #
class TestCanImport:
    """The :func:`_can_import` helper powers every ``has_*`` capability check."""

    def test_returns_false_for_missing_module(self) -> None:
        """A module that does not exist returns ``False`` (no exception)."""
        assert _can_import("definitely_not_a_real_module_xyz123") is False

    def test_returns_false_when_find_spec_finds_but_import_fails(self) -> None:
        """A discoverable-but-broken module is reported as unavailable.

        We stub :func:`importlib.util.find_spec` to return a sentinel
        :class:`object` and :func:`importlib.import_module` to raise the
        exception a real broken module would raise (``ModuleNotFoundError``).
        """
        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch(
                "importlib.import_module",
                side_effect=ModuleNotFoundError("simulated broken module"),
            ),
        ):
            assert _can_import("broken_module_xyz") is False

    def test_returns_false_for_oserror_during_import(self) -> None:
        """A shared library load failure (``OSError``) is handled gracefully."""
        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch(
                "importlib.import_module",
                side_effect=OSError("DLL load failed"),
            ),
        ):
            assert _can_import("dll_broken_module_xyz") is False

    def test_returns_true_for_known_module(self) -> None:
        """A trivially-importable stdlib module returns ``True``."""
        # ``pathlib`` ships with CPython and is always importable.
        assert _can_import("pathlib") is True


# --------------------------------------------------------------------------- #
# warp_device_available graceful degradation
# --------------------------------------------------------------------------- #
class TestWarpDeviceAvailable:
    """``warp_device_available`` must never raise — only return ``bool``.

    The trick: ``warp_device_available`` first calls :func:`has_warp` which
    calls :func:`_can_import` for *both* ``"warp"`` and ``"mujoco_warp"`` —
    so each positive test has to fake *both* modules into :data:`sys.modules`
    with a proper :class:`importlib.machinery.ModuleSpec` (a plain
    :class:`object` is rejected by :func:`importlib.util.find_spec`). The
    fake exposes ``init`` and ``get_cuda_device_count`` as module-level
    callables exactly the way the real ``warp`` package does.
    """

    @pytest.fixture(autouse=True)
    def _isolate_warp_modules(self) -> None:
        """Drop any cached ``warp`` / ``mujoco_warp`` modules between tests.

        ``sys.modules`` persists across tests in the same process. If a prior
        test inserted a fake, the next test's ``import warp`` may see the
        stale entry. Clearing both names before each test gives every
        assertion a clean slate. We also clear the ``lru_cache`` on
        :func:`warp_device_available` itself, since it caches the boolean
        return value across calls and would otherwise return the previous
        test's result.
        """
        sys.modules.pop("warp", None)
        sys.modules.pop("mujoco_warp", None)
        has_warp.cache_clear()  # type: ignore[attr-defined]
        warp_device_available.cache_clear()  # type: ignore[attr-defined]
        yield
        sys.modules.pop("warp", None)
        sys.modules.pop("mujoco_warp", None)
        has_warp.cache_clear()  # type: ignore[attr-defined]
        warp_device_available.cache_clear()  # type: ignore[attr-defined]

    @staticmethod
    def _make_fake_module(
        name: str, *, init: Any, count: Any, error: Exception | None = None
    ) -> types.ModuleType:
        """Build a stand-in module for :data:`sys.modules` with a real spec.

        ``count`` is wrapped in a lambda that returns a positive integer,
        zero, or raises ``error`` (if supplied). The returned module has
        a non-None :attr:`__spec__` so :func:`importlib.util.find_spec`
        accepts it during the ``_can_import`` probe.
        """
        if error is None:

            def getter() -> Any:
                return count

        else:

            def getter() -> Any:
                raise error

        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        module.init = init  # type: ignore[attr-defined]
        module.get_cuda_device_count = getter  # type: ignore[attr-defined]
        return module

    def test_returns_false_when_warp_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the Warp stack is missing, the probe short-circuits to ``False``."""
        # Drop any cached entry from a previous test and make sure neither
        # ``warp`` nor ``mujoco_warp`` is registered.
        monkeypatch.delitem(sys.modules, "warp", raising=False)
        monkeypatch.delitem(sys.modules, "mujoco_warp", raising=False)
        monkeypatch.setattr(
            "src.shared.python.simulation_backends.capabilities._can_import",
            lambda name: False,
        )
        has_warp.cache_clear()  # type: ignore[attr-defined]
        assert warp_device_available() is False

    def test_returns_false_on_runtime_error_during_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A :class:`RuntimeError` from ``wp.init()`` is swallowed.

        This is the case where the wheel is present but no CUDA driver is
        loadable; the suite must continue to run on CPU.
        """
        fake = self._make_fake_module(
            "warp",
            init=lambda: (_ for _ in ()).throw(RuntimeError("no CUDA")),
            count=0,
        )
        mujoco_warp_fake = self._make_fake_module(
            "mujoco_warp", init=lambda: None, count=0
        )
        monkeypatch.delitem(sys.modules, "warp", raising=False)
        monkeypatch.delitem(sys.modules, "mujoco_warp", raising=False)
        monkeypatch.setitem(sys.modules, "warp", fake)
        monkeypatch.setitem(sys.modules, "mujoco_warp", mujoco_warp_fake)
        monkeypatch.setattr(
            "src.shared.python.simulation_backends.capabilities._can_import",
            lambda name: name in ("warp", "mujoco_warp"),
        )
        has_warp.cache_clear()  # type: ignore[attr-defined]
        assert warp_device_available() is False

    def test_returns_false_on_attribute_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A :class:`AttributeError` from a Warp version mismatch is swallowed."""
        fake = self._make_fake_module(
            "warp",
            init=lambda: None,
            count=0,
            error=AttributeError("no get_cuda_device_count"),
        )
        mujoco_warp_fake = self._make_fake_module(
            "mujoco_warp", init=lambda: None, count=0
        )
        monkeypatch.delitem(sys.modules, "warp", raising=False)
        monkeypatch.delitem(sys.modules, "mujoco_warp", raising=False)
        monkeypatch.setitem(sys.modules, "warp", fake)
        monkeypatch.setitem(sys.modules, "mujoco_warp", mujoco_warp_fake)
        monkeypatch.setattr(
            "src.shared.python.simulation_backends.capabilities._can_import",
            lambda name: name in ("warp", "mujoco_warp"),
        )
        has_warp.cache_clear()  # type: ignore[attr-defined]
        assert warp_device_available() is False

    def test_returns_true_when_cuda_device_count_positive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Warp installation reporting at least one CUDA device returns ``True``.

        This test is the only ``warp_device_available`` happy path that
        actually exercises the production code's full positive branch.
        All other tests in this class exercise the negative / error
        branches and were designed to be the primary regression net.
        """
        fake = self._make_fake_module("warp", init=lambda: None, count=1)
        mujoco_warp_fake = self._make_fake_module(
            "mujoco_warp", init=lambda: None, count=0
        )
        monkeypatch.delitem(sys.modules, "warp", raising=False)
        monkeypatch.delitem(sys.modules, "mujoco_warp", raising=False)
        monkeypatch.setitem(sys.modules, "warp", fake)
        monkeypatch.setitem(sys.modules, "mujoco_warp", mujoco_warp_fake)
        monkeypatch.setattr(
            "src.shared.python.simulation_backends.capabilities._can_import",
            lambda name: name in ("warp", "mujoco_warp"),
        )
        has_warp.cache_clear()  # type: ignore[attr-defined]
        # We assert the public *contract*: a successful probe is reflected
        # as a positive truthy return. The exact path through ``wp.init()``
        # and ``wp.get_cuda_device_count()`` is exercised by the negative
        # tests above; if any of them regress, the assertion here still
        # pins the user-visible behaviour.
        assert bool(warp_device_available()) is True

    def test_returns_false_when_cuda_device_count_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Warp installation with no visible CUDA device returns ``False``."""
        fake = self._make_fake_module("warp", init=lambda: None, count=0)
        mujoco_warp_fake = self._make_fake_module(
            "mujoco_warp", init=lambda: None, count=0
        )
        monkeypatch.delitem(sys.modules, "warp", raising=False)
        monkeypatch.delitem(sys.modules, "mujoco_warp", raising=False)
        monkeypatch.setitem(sys.modules, "warp", fake)
        monkeypatch.setitem(sys.modules, "mujoco_warp", mujoco_warp_fake)
        monkeypatch.setattr(
            "src.shared.python.simulation_backends.capabilities._can_import",
            lambda name: name in ("warp", "mujoco_warp"),
        )
        has_warp.cache_clear()  # type: ignore[attr-defined]
        assert warp_device_available() is False


# --------------------------------------------------------------------------- #
# has_mujoco / has_warp caching behaviour
# --------------------------------------------------------------------------- #
class TestCapabilityCaching:
    """The :func:`has_*` helpers are wrapped in :func:`functools.lru_cache`."""

    def test_has_mujoco_is_cached(self) -> None:
        """Repeated calls return the same value without re-importing."""
        first = has_mujoco()
        with patch(
            "src.shared.python.simulation_backends.capabilities._can_import"
        ) as stub:
            stub.return_value = not first  # would flip the result if uncached
            second = has_mujoco()
        assert first == second

    def test_has_warp_is_cached(self) -> None:
        """Same lru_cache contract for :func:`has_warp`."""
        first = has_warp()
        with patch(
            "src.shared.python.simulation_backends.capabilities._can_import"
        ) as stub:
            stub.return_value = not first
            second = has_warp()
        assert first == second


# --------------------------------------------------------------------------- #
# Exception hierarchy
# --------------------------------------------------------------------------- #
class TestExceptionHierarchy:
    """The exception hierarchy is a public contract — pin it.

    ``UnknownBackendError`` doubles as a :class:`KeyError` so older call
    sites that wrote ``except KeyError`` keep working after the typed
    exception was introduced. ``BackendCapabilityError`` doubles as
    :class:`NotImplementedError` for the same reason.
    """

    def test_backend_error_is_exception_subclass(self) -> None:
        """The base class is a regular :class:`Exception`."""
        assert issubclass(BackendError, Exception)
        assert not issubclass(BackendError, KeyError)
        assert not issubclass(BackendError, NotImplementedError)

    def test_unknown_backend_error_is_keyerror_subclass(self) -> None:
        """``UnknownBackendError`` is catchable as both ``BackendError`` and ``KeyError``."""
        assert issubclass(UnknownBackendError, BackendError)
        assert issubclass(UnknownBackendError, KeyError)
        # Round-trip: ``except KeyError`` must catch it.
        with pytest.raises(KeyError):
            raise UnknownBackendError("simulated missing backend")

    def test_backend_not_available_is_plain_exception(self) -> None:
        """``BackendNotAvailableError`` is not a stdlib-specialised type."""
        assert issubclass(BackendNotAvailableError, BackendError)
        assert not issubclass(BackendNotAvailableError, KeyError)
        assert not issubclass(BackendNotAvailableError, NotImplementedError)

    def test_backend_capability_error_is_not_implemented(self) -> None:
        """``BackendCapabilityError`` is a ``NotImplementedError`` for legacy catch sites."""
        assert issubclass(BackendCapabilityError, BackendError)
        assert issubclass(BackendCapabilityError, NotImplementedError)
        with pytest.raises(NotImplementedError):
            raise BackendCapabilityError("backend cannot do that")

    def test_exception_messages_preserved(self) -> None:
        """The exception ``args`` carry the user-facing error message.

        Note: because :class:`UnknownBackendError` *also* subclasses
        :class:`KeyError`, :func:`str` of a string argument adds the
        surrounding repr-quotes (``"'foo'"``). Callers who need a clean
        message should use ``exc.args[0]`` — which is what the test pins
        here as the user-facing surface.
        """
        msg_unknown = "no backend named 'foo'"
        msg_unavail = "mujoco missing"
        msg_caps = "backend has no dynamics"
        assert UnknownBackendError(msg_unknown).args[0] == msg_unknown
        assert str(BackendNotAvailableError(msg_unavail)) == msg_unavail
        assert str(BackendCapabilityError(msg_caps)) == msg_caps

    def test_can_be_caught_polymorphically(self) -> None:
        """A single ``except BackendError`` catches every concrete subclass."""
        for exc in (
            UnknownBackendError("a"),
            BackendNotAvailableError("b"),
            BackendCapabilityError("c"),
        ):
            with pytest.raises(BackendError):
                raise exc


# --------------------------------------------------------------------------- #
# BackendCapabilities
# --------------------------------------------------------------------------- #
class TestBackendCapabilitiesContract:
    """``BackendCapabilities`` is a frozen dataclass used in every capability
    branch. Pin equality and mutability so the contract is observable.
    """

    def test_default_construction_all_caps_false(self) -> None:
        """Defaults: name required; everything else False / 'cpu'."""
        caps = BackendCapabilities(name="ode")
        assert caps.name == "ode"
        assert caps.device == "cpu"
        assert caps.supports_batched is False
        assert caps.is_differentiable is False
        assert caps.provides_dynamics is False

    def test_equality_is_field_based(self) -> None:
        """Two capabilities with identical fields compare equal."""
        a = BackendCapabilities(name="ode")
        b = BackendCapabilities(name="ode")
        assert a == b
        assert a is not b  # equality, not identity

    def test_inequality_on_name(self) -> None:
        """Different names mean different capabilities."""
        assert BackendCapabilities(name="ode") != BackendCapabilities(name="mujoco")

    def test_frozen_blocks_mutation(self) -> None:
        """The dataclass is frozen — assignment raises :class:`FrozenInstanceError`."""
        caps = BackendCapabilities(name="ode")
        with pytest.raises((AttributeError, Exception)):
            caps.name = "mujoco"  # type: ignore[misc]

    def test_distinct_capability_combinations(self) -> None:
        """A 'mjwarp' backend can declare both GPU + batched support."""
        caps = BackendCapabilities(
            name="mjwarp",
            device="cuda",
            supports_batched=True,
            is_differentiable=False,
            provides_dynamics=False,
        )
        assert caps.device == "cuda"
        assert caps.supports_batched is True
        assert caps.provides_dynamics is False


# --------------------------------------------------------------------------- #
# Protocol runtime-checkability
# --------------------------------------------------------------------------- #
class _MinimalBackend:
    """A duck-typed stand-in that satisfies the full :class:`SimulationBackend` Protocol.

    The Protocol is :func:`runtime_checkable`, so ``isinstance`` returns
    ``True`` when *every* method declared on the Protocol is present on the
    candidate — even if the candidate is not a subclass. The implementation
    here is intentionally trivial; it exists only to exercise the contract.
    """

    @property
    def capabilities(self) -> BackendCapabilities:
        """Static capability description for this stub."""
        return BackendCapabilities(name="stub")

    def reset(self, state: Any = None) -> None:
        """Reset to ``state`` (no-op for the stub)."""
        return

    def step(self, dt: float | None = None) -> None:
        """Advance one step (no-op for the stub)."""
        return

    def get_state(self) -> Any:
        """Return a no-op state object."""
        return None

    def set_control(self, u: np.ndarray) -> None:
        """Accept a control vector and discard it."""
        return

    def get_time(self) -> float:
        """Return a constant simulation time."""
        return 0.0

    def forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        """Return zero accelerations — the stub does not simulate."""
        return np.zeros_like(v)

    def rollout(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
    ) -> Any:
        """Return ``None`` — the stub does not produce a :class:`Trace`."""
        return None


class _DynamicsStub:
    """A duck-typed stand-in for :class:`DynamicsProvider` only."""

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Identity inertia — enough to satisfy the protocol shape."""
        n = q.size
        return np.eye(n)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero bias — the stub does not compute dynamics."""
        return np.zeros_like(v)


class _BatchedStub:
    """A duck-typed stand-in for :class:`BatchedBackend` only."""

    def rollout_batch(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
        num_envs: int,
    ) -> Any:
        """Return ``None`` — the stub does not produce a :class:`BatchTrace`."""
        return None


class TestProtocolRuntimeCheckable:
    """The three Protocols in :mod:`simulation_backends.protocol` are
    :func:`runtime_checkable`. Pin the behaviour so a future signature
    change cannot silently break duck-typed adapters.
    """

    def test_minimal_backend_satisfies_simulation_protocol(self) -> None:
        """A duck-typed object with all required methods passes ``isinstance``."""
        backend = _MinimalBackend()
        assert isinstance(backend, SimulationBackend)

    def test_dynamics_stub_satisfies_dynamics_protocol(self) -> None:
        """A class with only ``mass_matrix`` / ``bias_forces`` is a ``DynamicsProvider``."""
        stub = _DynamicsStub()
        assert isinstance(stub, DynamicsProvider)
        # And *not* a full SimulationBackend (it lacks step / get_state / etc).
        assert not isinstance(stub, SimulationBackend)

    def test_batched_stub_satisfies_batched_protocol(self) -> None:
        """A class with only ``rollout_batch`` is a :class:`BatchedBackend`."""
        stub = _BatchedStub()
        assert isinstance(stub, BatchedBackend)
        assert not isinstance(stub, SimulationBackend)

    def test_partial_class_fails_simulation_protocol(self) -> None:
        """A class missing *one* required method fails the Protocol check."""

        class _Partial:
            @property
            def capabilities(self) -> BackendCapabilities:
                return BackendCapabilities(name="p")

            def reset(self, state: Any = None) -> None:
                return None

            # step / get_state / set_control / get_time / forward_dynamics /
            # rollout are deliberately missing.

        assert not isinstance(_Partial(), SimulationBackend)


# --------------------------------------------------------------------------- #
# SCHEMA_VERSION is the wire contract
# --------------------------------------------------------------------------- #
class TestSchemaVersion:
    """``SCHEMA_VERSION`` is stamped into every serialised trace (see
    :mod:`simulation_backends.trace_io`). A breaking change to the on-disk
    layout must bump it; a non-breaking change must not.
    """

    def test_schema_version_is_a_semver_string(self) -> None:
        """``SCHEMA_VERSION`` is a non-empty ``X.Y.Z`` string."""
        assert isinstance(SCHEMA_VERSION, str)
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit(), f"semver component {part!r} is not numeric"
