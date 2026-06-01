"""TDD tests for engine route/frontend alignment (issue #2482).

Five bugs:
1. engines.py probe and load routes hard-code '/api/' prefix causing double-prefix
   when server registers with prefix='/api'.
2. useEngineStore.ts unloadEngine only mutates client state; never calls backend.
3. Simulation.tsx shows success toast after requestLoad() regardless of error state.
4. useEngineCapabilities.ts uses '/engines/...' (no /api/ prefix) unlike the rest of
   the frontend which uses '/api/engines/...'.
5. ParameterPanel.tsx defines 'myosim' defaults but the engine identifier is 'myosuite'.
"""

from __future__ import annotations

from pathlib import Path

_ENGINES_PY = Path("src/api/routes/engines.py")
_ENGINE_STORE = Path("ui/src/stores/useEngineStore.ts")
_SIMULATION_TSX = Path("ui/src/pages/Simulation.tsx")
_CAPABILITIES_TS = Path("ui/src/api/useEngineCapabilities.ts")
_PARAM_PANEL = Path("ui/src/components/simulation/ParameterPanel.tsx")


class TestEnginesPyNoHardcodedApiPrefix:
    """engines.py route decorators must not embed /api/ — server injects the prefix."""

    def _source(self) -> str:
        return _ENGINES_PY.read_text(encoding="utf-8")

    def test_probe_route_has_no_api_prefix(self) -> None:
        """The probe route must not start with /api/."""
        source = self._source()
        lines = source.splitlines()
        bad = [
            line
            for line in lines
            if '"/api/engines/' in line
            and "probe" in line
            and not line.strip().startswith("#")
        ]
        assert not bad, (
            "engines.py probe route embeds '/api/' prefix. "
            "The server adds '/api' via register_routes, so the route path would "
            "become /api/api/engines/.../probe. Fix: change to '/engines/{name}/probe'.\n"
            "Offending lines:\n" + "\n".join(bad)
        )

    def test_lazy_load_route_has_no_api_prefix(self) -> None:
        """The lazy-load route must not start with /api/."""
        source = self._source()
        lines = source.splitlines()
        bad = [
            line
            for line in lines
            if '"/api/engines/' in line
            and "load" in line
            and not line.strip().startswith("#")
        ]
        assert not bad, (
            "engines.py load route embeds '/api/' prefix. "
            "Fix: change to '/engines/{name}/load'.\n"
            "Offending lines:\n" + "\n".join(bad)
        )


class TestEngineStoreCallsBackendUnload:
    """useEngineStore.ts unloadEngine must call the backend unload endpoint."""

    def _source(self) -> str:
        return _ENGINE_STORE.read_text(encoding="utf-8")

    def test_unload_engine_calls_fetch(self) -> None:
        """unloadEngine must call fetch to hit the backend /engines/{type}/unload route."""
        source = self._source()
        lines = source.splitlines()
        # Find the unloadEngine function body
        in_unload = False
        unload_body: list[str] = []
        brace_depth = 0
        for line in lines:
            if "unloadEngine:" in line or "unloadEngine =" in line:
                in_unload = True
            if in_unload:
                unload_body.append(line)
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0 and len(unload_body) > 1:
                    break

        unload_str = "\n".join(unload_body)
        assert "fetch" in unload_str.lower() or "axios" in unload_str, (
            "useEngineStore.ts unloadEngine does not call fetch() to notify the backend. "
            "Backend exposes POST /engines/{type}/unload. "
            "Fix: add a fetch call in unloadEngine before mutating client state."
        )


class TestSimulationSuccessToastOnlyOnSuccess:
    """Simulation.tsx must not show success toast when requestLoad() encounters an error."""

    def _source(self) -> str:
        return _SIMULATION_TSX.read_text(encoding="utf-8")

    def test_success_toast_guarded_by_error_check(self) -> None:
        """showSuccess must not be called unconditionally after requestLoad()."""
        source = self._source()
        lines = source.splitlines()
        # Find handleLoadEngine block
        in_handler = False
        handler_body: list[str] = []
        for line in lines:
            if "handleLoadEngine" in line and "useCallback" in line:
                in_handler = True
            if in_handler:
                handler_body.append(line)
                if line.strip() == ")," and len(handler_body) > 3:
                    break

        # After fix: success toast must only appear if requestLoad succeeded
        # (check for error/catch before showSuccess, or check store state)
        success_line_idx = next(
            (i for i, ln in enumerate(handler_body) if "showSuccess" in ln), None
        )
        if success_line_idx is None:
            return  # No success toast — no issue

        # Check if there's a conditional or try/catch guarding showSuccess
        pre_success = "\n".join(handler_body[:success_line_idx])
        has_guard = (
            "catch" in pre_success
            or "if" in pre_success
            or "error" in pre_success.lower()
            or "try" in pre_success
        )
        assert has_guard, (
            "Simulation.tsx calls showSuccess unconditionally after requestLoad(). "
            "If requestLoad sets an error state, the success toast still fires. "
            "Fix: wrap in try/catch or check the engine's loadState after calling requestLoad."
        )


class TestCapabilitiesHookUsesApiPrefix:
    """useEngineCapabilities.ts must use /api/engines/... like the rest of the frontend."""

    def _source(self) -> str:
        return _CAPABILITIES_TS.read_text(encoding="utf-8")

    def test_capabilities_url_has_api_prefix(self) -> None:
        """fetch URL must start with /api/engines/, not /engines/."""
        source = self._source()
        lines = source.splitlines()
        bad_url_lines = [
            line
            for line in lines
            if "fetch(`/engines/" in line and not line.strip().startswith("//")
        ]
        assert not bad_url_lines, (
            "useEngineCapabilities.ts uses /engines/ URL without /api/ prefix. "
            "Other frontend calls use /api/engines/. Fix: change to /api/engines/.\n"
            "Offending lines:\n" + "\n".join(bad_url_lines)
        )


class TestParameterPanelMyosuiteDefaults:
    """ParameterPanel.tsx must define defaults for 'myosuite', not 'myosim'."""

    def _source(self) -> str:
        return _PARAM_PANEL.read_text(encoding="utf-8")

    def test_myosuite_key_in_defaults(self) -> None:
        """ENGINE_DEFAULTS must have 'myosuite' key (the actual engine identifier)."""
        source = self._source()
        assert "myosuite" in source, (
            "ParameterPanel.tsx ENGINE_DEFAULTS does not define 'myosuite'. "
            "The engine registry uses name 'myosuite'. "
            "Fix: add/rename 'myosuite' key in ENGINE_DEFAULTS."
        )

    def test_no_standalone_myosim_key(self) -> None:
        """ENGINE_DEFAULTS must not have a bare 'myosim' key without 'myosuite'."""
        source = self._source()
        lines = source.splitlines()
        # Find myosim key in defaults object (not in a comment or string like 'myosim...')
        myosim_only_lines = [
            line
            for line in lines
            if "myosim:" in line
            and "myosuite" not in line
            and not line.strip().startswith("//")
        ]
        # If myosuite also exists, myosim is just an alias — OK
        # If myosuite does NOT exist, myosim is the only entry — NOT OK
        if "myosuite" not in source:
            assert not myosim_only_lines, (
                "ParameterPanel.tsx only has 'myosim' defaults, not 'myosuite'. "
                "The engine is registered as 'myosuite'. Rename the key to 'myosuite'.\n"
                "Offending lines:\n" + "\n".join(myosim_only_lines)
            )
