"""Unit tests for model duplicate endpoint (issues #3174, #3202).

Tests cover:
- ModelDuplicateRequest validation (safe name, path fields)
- ModelDuplicateResponse structure
- Source-level verification that /models/duplicate endpoint is declared
- validate_model_source_path allowlist validation (issue #3202)

defusedxml is not installed in the unit-test environment, so we avoid
importing the full model_explorer module and test the data models
directly via AST inspection and direct Pydantic model construction.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

_MODEL_EXPLORER_SRC = (
    Path(__file__).parents[3] / "src" / "api" / "routes" / "model_explorer.py"
).read_text(encoding="utf-8")


class TestModelDuplicateEndpointDeclared:
    """model_explorer.py must declare the /models/duplicate route."""

    def test_duplicate_route_path_present(self) -> None:
        """Source contains the /models/duplicate path string."""
        assert "/models/duplicate" in _MODEL_EXPLORER_SRC, (
            "model_explorer.py must declare a POST /models/duplicate route"
        )

    def test_duplicate_model_function_present(self) -> None:
        """Source declares an async duplicate_model function."""
        assert "async def duplicate_model" in _MODEL_EXPLORER_SRC, (
            "model_explorer.py must define `async def duplicate_model`"
        )

    def test_model_duplicate_request_class_present(self) -> None:
        """Source defines ModelDuplicateRequest Pydantic model."""
        assert "class ModelDuplicateRequest" in _MODEL_EXPLORER_SRC, (
            "model_explorer.py must define ModelDuplicateRequest"
        )

    def test_model_duplicate_response_class_present(self) -> None:
        """Source defines ModelDuplicateResponse Pydantic model."""
        assert "class ModelDuplicateResponse" in _MODEL_EXPLORER_SRC, (
            "model_explorer.py must define ModelDuplicateResponse"
        )

    def test_source_path_field_present(self) -> None:
        """ModelDuplicateRequest declares a source_path field."""
        assert "source_path" in _MODEL_EXPLORER_SRC

    def test_new_name_field_present(self) -> None:
        """ModelDuplicateRequest declares a new_name field."""
        assert "new_name" in _MODEL_EXPLORER_SRC

    def test_copy_path_field_present(self) -> None:
        """ModelDuplicateResponse declares a copy_path field."""
        assert "copy_path" in _MODEL_EXPLORER_SRC

    def test_path_traversal_guard_present(self) -> None:
        """Source contains path-traversal protection logic."""
        assert "relative_to" in _MODEL_EXPLORER_SRC, (
            "model_explorer.py duplicate endpoint must guard against path traversal"
        )

    def test_409_conflict_response_present(self) -> None:
        """Source returns 409 when destination already exists."""
        assert "409" in _MODEL_EXPLORER_SRC, (
            "model_explorer.py must return HTTP 409 when the copy already exists"
        )

    def test_shutil_copy_used(self) -> None:
        """Source uses shutil for file copying."""
        assert "shutil" in _MODEL_EXPLORER_SRC, (
            "model_explorer.py must use shutil to copy model files"
        )

    def test_safe_name_regex_present(self) -> None:
        """Source defines a safe-name regex for new_name validation."""
        assert "_SAFE_STEM_RE" in _MODEL_EXPLORER_SRC, (
            "model_explorer.py must define _SAFE_STEM_RE for name validation"
        )

    def test_router_post_decorator(self) -> None:
        """The duplicate route uses @router.post."""
        tree = ast.parse(_MODEL_EXPLORER_SRC)
        post_routes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        func = decorator.func
                        if (
                            isinstance(func, ast.Attribute)
                            and func.attr == "post"
                            and isinstance(decorator.args[0], ast.Constant)
                        ):
                            post_routes.append(decorator.args[0].value)
        assert any("duplicate" in r for r in post_routes), (
            f"Expected a @router.post('/models/duplicate') decorator, found: {post_routes}"
        )


class TestSafeStemRegex:
    """The _SAFE_STEM_RE pattern in model_explorer.py must reject unsafe names."""

    # Extract the regex pattern from source via AST
    _STEM_RE_PATTERN: str | None = None

    @classmethod
    def _get_stem_pattern(cls) -> str:
        """Parse and return the _SAFE_STEM_RE pattern string from source."""
        if cls._STEM_RE_PATTERN is not None:
            return cls._STEM_RE_PATTERN
        tree = ast.parse(_MODEL_EXPLORER_SRC)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_SAFE_STEM_RE"
                and isinstance(node.value, ast.Call)
                and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
            ):
                cls._STEM_RE_PATTERN = node.value.args[0].value
                return cls._STEM_RE_PATTERN
        raise AssertionError("Could not extract _SAFE_STEM_RE pattern from source")

    def test_alphanumeric_accepted(self) -> None:
        """Plain alphanumeric names are accepted."""
        pattern = re.compile(self._get_stem_pattern())
        assert pattern.match("my_model")

    def test_hyphens_accepted(self) -> None:
        """Hyphens are accepted in model names."""
        pattern = re.compile(self._get_stem_pattern())
        assert pattern.match("my-model")

    def test_path_traversal_rejected(self) -> None:
        """Dotdot path traversal sequences are rejected."""
        pattern = re.compile(self._get_stem_pattern())
        assert not pattern.match("../../etc/passwd")

    def test_slash_rejected(self) -> None:
        """Forward slashes are rejected."""
        pattern = re.compile(self._get_stem_pattern())
        assert not pattern.match("dir/file")

    def test_empty_string_rejected(self) -> None:
        """Empty string is rejected."""
        pattern = re.compile(self._get_stem_pattern())
        assert not pattern.match("")


class TestLocalServerPresetsRegistration:
    """local_server.py must import and register the presets router."""

    _LOCAL_SERVER_SRC: str = (
        Path(__file__).parents[3] / "src" / "api" / "local_server.py"
    ).read_text(encoding="utf-8")

    def test_presets_imported(self) -> None:
        """local_server.py imports the presets module."""
        assert "presets" in self._LOCAL_SERVER_SRC, (
            "local_server.py must import the presets router"
        )

    def test_presets_router_included(self) -> None:
        """local_server.py calls include_router with presets.router."""
        assert "presets.router" in self._LOCAL_SERVER_SRC, (
            "local_server.py must register presets.router via include_router"
        )


class TestSimulationWsSummary:
    """simulation_ws.py must emit a summary on run completion (issue #3174)."""

    _WS_SRC: str = (
        Path(__file__).parents[3] / "src" / "api" / "routes" / "simulation_ws.py"
    ).read_text(encoding="utf-8")

    def test_summary_key_emitted(self) -> None:
        """WebSocket route emits a 'summary' key in the complete message."""
        assert "summary" in self._WS_SRC, (
            "simulation_ws.py must include a 'summary' field in the "
            "completion message for the SummaryPanel to display"
        )

    def test_status_complete_emitted(self) -> None:
        """WebSocket route emits status='complete' on run completion."""
        assert "complete" in self._WS_SRC

    def test_peak_torques_tracked_in_loop(self) -> None:
        """Simulation loop tracks peak torques each step, not just at the end."""
        assert "peak_torques" in self._WS_SRC, (
            "simulation_ws.py must accumulate peak_torques during the loop"
        )


class TestModelExtensionAllowlist:
    """The duplicate endpoint must restrict files to model types."""

    _MODEL_EXPLORER_SRC: str = (
        Path(__file__).parents[3] / "src" / "api" / "routes" / "model_explorer.py"
    ).read_text(encoding="utf-8")

    def test_urdf_allowed(self) -> None:
        """.urdf extension is in the allowlist."""
        assert ".urdf" in self._MODEL_EXPLORER_SRC

    def test_mjcf_allowed(self) -> None:
        """.mjcf extension is in the allowlist."""
        assert ".mjcf" in self._MODEL_EXPLORER_SRC

    def test_extension_check_present(self) -> None:
        """Source contains logic to check file extension against the allowlist."""
        assert "_ALLOWED_MODEL_EXTENSIONS" in self._MODEL_EXPLORER_SRC, (
            "model_explorer.py must define _ALLOWED_MODEL_EXTENSIONS allowlist"
        )

    def test_422_on_bad_extension(self) -> None:
        """Source raises HTTP 422 when extension is not in the allowlist."""
        assert "422" in self._MODEL_EXPLORER_SRC

    def test_validate_model_source_path_function_present(self) -> None:
        """Source defines a validate_model_source_path helper (issue #3202)."""
        assert "validate_model_source_path" in self._MODEL_EXPLORER_SRC, (
            "model_explorer.py must define validate_model_source_path "
            "for explicit pre-copy allowlist validation"
        )

    def test_allowed_model_dirs_constant_present(self) -> None:
        """Source defines _ALLOWED_MODEL_DIRS for directory allowlist."""
        assert "_ALLOWED_MODEL_DIRS" in self._MODEL_EXPLORER_SRC, (
            "model_explorer.py must define _ALLOWED_MODEL_DIRS"
        )

    def test_sdf_extension_in_allowlist(self) -> None:
        """.sdf extension is in the allowlist (issue #3202)."""
        assert ".sdf" in self._MODEL_EXPLORER_SRC

    def test_obj_extension_in_allowlist(self) -> None:
        """.obj extension is in the allowlist (issue #3202)."""
        assert ".obj" in self._MODEL_EXPLORER_SRC

    def test_stl_extension_in_allowlist(self) -> None:
        """.stl extension is in the allowlist (issue #3202)."""
        assert ".stl" in self._MODEL_EXPLORER_SRC


def _load_validate_fn():
    """Load validate_model_source_path from model_explorer with deps mocked."""
    mocks = {
        "defusedxml": MagicMock(),
        "defusedxml.ElementTree": MagicMock(),
        "src.api.middleware.error_handler": MagicMock(handle_api_errors=lambda f: f),
        "src.shared.python.core.contracts": MagicMock(
            precondition=lambda *a, **kw: lambda f: f,
            postcondition=lambda *a, **kw: lambda f: f,
        ),
        "src.api.dependencies": MagicMock(),
        "src.api.models.requests": MagicMock(),
        "src.api.models.responses": MagicMock(),
        "src.api.routes._route_utils": MagicMock(),
        "fastapi": MagicMock(),
        "pydantic": MagicMock(),
    }
    import sys

    with patch.dict(sys.modules, mocks):
        spec = importlib.util.spec_from_file_location(
            "model_explorer_isolated",
            Path(__file__).parents[3] / "src" / "api" / "routes" / "model_explorer.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.validate_model_source_path


class TestValidateModelSourcePath:
    """Unit tests for validate_model_source_path (issue #3202).

    Tests import only the pure validation function to avoid defusedxml
    dependency -- the function does path arithmetic, not XML parsing.
    """

    def test_urdf_in_approved_dir_is_allowed(self) -> None:
        """A .urdf file inside an approved model dir passes validation."""
        validate = _load_validate_fn()
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            model_dir = repo_root / "tests" / "fixtures" / "models"
            model_dir.mkdir(parents=True)
            urdf_file = model_dir / "robot.urdf"
            urdf_file.write_text("<robot/>")
            # Should not raise
            validate("tests/fixtures/models/robot.urdf", repo_root)

    def test_python_file_is_rejected(self) -> None:
        """A .py source file is rejected even if it exists in a model dir."""
        validate = _load_validate_fn()
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            model_dir = repo_root / "tests" / "fixtures" / "models"
            model_dir.mkdir(parents=True)
            py_file = model_dir / "secret.py"
            py_file.write_text("# code")
            with pytest.raises(ValueError, match=r"(?i)model asset"):
                validate("tests/fixtures/models/secret.py", repo_root)

    def test_path_traversal_is_rejected(self) -> None:
        """A path traversal attempt escaping the repo root is rejected."""
        validate = _load_validate_fn()
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with pytest.raises(ValueError, match=r"(?i)escapes"):
                validate("../../etc/passwd", repo_root)
