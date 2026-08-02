# ruff: noqa: E501
"""ModelGenerationAPI class with all route handler methods.

This module contains the main API class that registers routes and implements
handlers for URDF generation, conversion, validation, inertia, library, and
editor operations.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .rest_api_types import APIRequest, APIResponse, HTTPMethod, Route

logger = logging.getLogger(__name__)
MAX_MESH_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MiB — closes #2479
ALLOWED_MESH_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".dae", ".glb", ".gltf"}


class ModelGenerationAPI:
    """
    REST API for model generation operations.

    Provides endpoints for:
    - URDF generation (parametric, humanoid)
    - Format conversion (SimScape, MJCF)
    - Model validation
    - Library operations
    - Inertia calculation

    Example with Flask:
        from flask import Flask, request, jsonify
        from model_generation.api import ModelGenerationAPI, FlaskAdapter

        app = Flask(__name__)
        api = ModelGenerationAPI()
        FlaskAdapter(api).register(app)

    Example with FastAPI:
        from fastapi import FastAPI
        from model_generation.api import ModelGenerationAPI, FastAPIAdapter

        app = FastAPI()
        api = ModelGenerationAPI()
        FastAPIAdapter(api).register(app)
    """

    def __init__(self, prefix: str = "/api/v1") -> None:
        """
        Initialize API.

        Args:
            prefix: URL prefix for all routes
        """
        if prefix is None:
            raise ValueError("prefix must be provided")
        self.prefix = prefix
        self._routes: list[Route] = []
        self._register_routes()

    def _register_core_routes(self) -> None:
        """Register health, generation, conversion, validation, and parsing routes."""
        self.add_route(HTTPMethod.GET, "/health", self.health_check, "Health check")
        self.add_route(HTTPMethod.GET, "/info", self.get_api_info, "API information")

        self.add_route(
            HTTPMethod.POST,
            "/generate/humanoid",
            self.generate_humanoid,
            "Generate humanoid URDF",
            ["generation"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/generate/from-params",
            self.generate_from_params,
            "Generate URDF from parameters",
            ["generation"],
        )

        self.add_route(
            HTTPMethod.POST,
            "/convert/simscape-to-urdf",
            self.convert_simscape_to_urdf,
            "Convert SimScape to URDF",
            ["conversion"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/convert/mjcf-to-urdf",
            self.convert_mjcf_to_urdf,
            "Convert MJCF to URDF",
            ["conversion"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/convert/urdf-to-mjcf",
            self.convert_urdf_to_mjcf,
            "Convert URDF to MJCF",
            ["conversion"],
        )

        self.add_route(
            HTTPMethod.POST,
            "/validate",
            self.validate_urdf,
            "Validate URDF content",
            ["validation"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/parse",
            self.parse_urdf,
            "Parse URDF and return structure",
            ["parsing"],
        )

    def _register_inertia_and_library_routes(self) -> None:
        """Register inertia calculation and library management routes."""
        self.add_route(
            HTTPMethod.POST,
            "/inertia/calculate",
            self.calculate_inertia,
            "Calculate inertia for shape",
            ["inertia"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/inertia/from-mesh",
            self.inertia_from_mesh,
            "Calculate inertia from mesh file",
            ["inertia"],
        )

        self.add_route(
            HTTPMethod.GET,
            "/library/models",
            self.library_list_models,
            "List available models",
            ["library"],
        )
        self.add_route(
            HTTPMethod.GET,
            "/library/models/{model_id}",
            self.library_get_model,
            "Get model details",
            ["library"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/library/models",
            self.library_add_model,
            "Add model to library",
            ["library"],
        )
        self.add_route(
            HTTPMethod.DELETE,
            "/library/models/{model_id}",
            self.library_remove_model,
            "Remove model from library",
            ["library"],
        )
        self.add_route(
            HTTPMethod.GET,
            "/library/models/{model_id}/download",
            self.library_download_model,
            "Download model URDF",
            ["library"],
        )

    def _register_editor_routes(self) -> None:
        """Register editor-related routes."""
        self.add_route(
            HTTPMethod.POST,
            "/editor/compose",
            self.compose_models,
            "Compose model from multiple sources",
            ["editor"],
        )
        self.add_route(
            HTTPMethod.POST,
            "/editor/diff",
            self.diff_urdfs,
            "Compare two URDF files",
            ["editor"],
        )

    def _register_routes(self) -> None:
        """Register all API routes."""
        self._register_core_routes()
        self._register_inertia_and_library_routes()
        self._register_editor_routes()

    def add_route(
        self,
        method: HTTPMethod,
        path: str,
        handler: Callable[[APIRequest], APIResponse],
        description: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Add a route to the API."""
        self._routes.append(
            Route(
                method=method,
                path=self.prefix + path,
                handler=handler,
                description=description,
                tags=tags or [],
            )
        )

    def get_routes(self) -> list[Route]:
        """Get all registered routes."""
        return self._routes

    def _add_security_headers(self, response: APIResponse) -> None:
        """Add security headers to response."""
        if response is None:
            raise ValueError("response must be provided")
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

    def handle_request(self, request: APIRequest) -> APIResponse:
        """Handle an API request."""
        # Find matching route
        if request is None:
            raise ValueError("request must be provided")
        for route in self._routes:
            if route.method != request.method:
                continue

            # Simple path matching (handles {param} patterns)
            route_parts = route.path.split("/")
            request_parts = request.path.split("/")

            if len(route_parts) != len(request_parts):
                continue

            params = {}
            match = True
            for rp, reqp in zip(route_parts, request_parts, strict=False):
                if rp.startswith("{") and rp.endswith("}"):
                    param_name = rp[1:-1]
                    params[param_name] = reqp
                elif rp != reqp:
                    match = False
                    break

            if match:
                # Add path params to query params
                request.query_params.update(params)
                try:
                    response = route.handler(request)
                    self._add_security_headers(response)
                    return response
                except Exception:  # noqa: BLE001
                    logger.exception("Error handling request")
                    response = APIResponse.error("Internal server error", 500)
                    self._add_security_headers(response)
                    return response

        response = APIResponse.not_found(
            f"No route for {request.method.value} {request.path}"
        )
        self._add_security_headers(response)
        return response

    # ============================================================
    # Health/Info Handlers
    # ============================================================

    @staticmethod
    def _extract_text_content(
        request: APIRequest,
        body: dict[str, Any],
        missing_message: str,
        *,
        body_key: str = "content",
        prefer_file: bool = False,
    ) -> tuple[str | None, APIResponse | None, str | None]:
        """Extract text from a JSON body field or uploaded file."""

        def _decode_file() -> tuple[str | None, APIResponse | None, str | None]:
            try:
                content = request.files["file"].decode("utf-8")
            except UnicodeDecodeError:
                return (
                    None,
                    APIResponse.error("Uploaded file must be valid UTF-8 text", 422),
                    "file",
                )
            if not content:
                return None, APIResponse.error(missing_message), "file"
            return content, None, "file"

        if prefer_file and "file" in request.files:
            return _decode_file()

        if body_key in body and body[body_key]:
            content = body[body_key]
            if not isinstance(content, str):
                return (
                    None,
                    APIResponse.error(f"'{body_key}' must be a string"),
                    "body",
                )
            return content, None, "body"

        if "file" in request.files:
            return _decode_file()

        return None, APIResponse.error(missing_message), None

    @staticmethod
    def _require_positive_number(value: Any, field_name: str) -> float:
        """Return *value* as a positive float or raise ValueError."""
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{field_name}' must be a positive number") from exc
        if number <= 0:
            raise ValueError(f"'{field_name}' must be a positive number")
        return number

    @staticmethod
    def _validate_operation_fields(
        op: dict[str, Any], op_type: str, required: tuple[str, ...]
    ) -> APIResponse | None:
        missing = [field for field in required if field not in op or op[field] is None]
        if missing:
            return APIResponse.error(
                f"Operation '{op_type}' missing required field(s): {', '.join(missing)}"
            )
        return None

    def health_check(self, request: APIRequest) -> APIResponse:
        """Health check endpoint."""
        return APIResponse.ok({"status": "healthy", "service": "model_generation"})

    def get_api_info(self, request: APIRequest) -> APIResponse:
        """Get API information."""
        return APIResponse.ok(
            {
                "name": "Model Generation API",
                "version": "1.0.0",
                "description": "REST API for URDF generation, conversion, and manipulation",
                "endpoints": [
                    {
                        "method": r.method.value,
                        "path": r.path,
                        "description": r.description,
                        "tags": r.tags,
                    }
                    for r in self._routes
                ],
            }
        )

    # ============================================================
    # Generation Handlers
    # ============================================================

    def generate_humanoid(self, request: APIRequest) -> APIResponse:
        """Generate humanoid URDF."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.builders.parametric_builder import (
            ParametricBuilder,
        )

        body = request.body or {}
        robot_name = body.get("name", "humanoid")
        height = body.get("height", 1.7)
        mass = body.get("mass", 70.0)

        builder = ParametricBuilder(robot_name=robot_name)

        # Apply parameters including proportions if provided
        proportions = body.get("proportions", {})
        builder.set_parameters(height_m=height, mass_kg=mass, **proportions)

        builder.add_humanoid_segments()
        result = builder.build()

        if not result.success:
            return APIResponse.error(result.error_message or "Build failed")

        urdf_string = result.urdf_xml

        # Return as file or JSON based on query param
        if request.query_params.get("download") == "true":
            return APIResponse.file(urdf_string, f"{robot_name}.urdf")

        return APIResponse.ok(
            {
                "robot_name": robot_name,
                "links": len(result.links),
                "joints": len(result.joints),
                "urdf": urdf_string,
            }
        )

    def generate_from_params(self, request: APIRequest) -> APIResponse:
        """Generate URDF from detailed parameters."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.builders.manual_builder import ManualBuilder
        from shared.python.model_generation.core.types import (
            Joint,
            Link,
        )

        body = request.body or {}

        if "links" not in body:
            return APIResponse.error("Missing 'links' in request body")

        robot_name = body.get("name", "robot")
        builder = ManualBuilder(robot_name=robot_name)

        # Add links
        for link_data in body.get("links", []):
            link = Link.from_dict(link_data)
            builder.add_link(link)

        # Add joints
        for joint_data in body.get("joints", []):
            joint = Joint.from_dict(joint_data)
            builder.add_joint(joint)

        result = builder.build()

        if not result.success:
            return APIResponse.error(result.error_message or "Build failed")

        urdf_string = result.urdf_xml

        if request.query_params.get("download") == "true":
            return APIResponse.file(urdf_string, f"{robot_name}.urdf")

        return APIResponse.ok(
            {
                "robot_name": robot_name,
                "links": len(result.links),
                "joints": len(result.joints),
                "urdf": urdf_string,
            }
        )

    # ============================================================
    # Conversion Handlers
    # ============================================================

    def convert_simscape_to_urdf(self, request: APIRequest) -> APIResponse:
        """Convert SimScape MDL/SLX to URDF."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.converters.simscape import (
            ConversionConfig,
            SimscapeToURDFConverter,
        )

        body = request.body or {}

        content, error, source = self._extract_text_content(
            request,
            body,
            "Missing model content or file",
            prefer_file=True,
        )
        if error:
            return error

        format_type = body.get("format", "mdl")
        if (
            source == "file"
            and content
            and (content.strip().startswith("<?xml") or content.strip().startswith("<"))
        ):
            format_type = "xml"

        robot_name = body.get("robot_name", "converted_robot")
        config = ConversionConfig(robot_name=robot_name)

        converter = SimscapeToURDFConverter(config)
        result = converter.convert_string(content, format_type)

        if not result.success:
            return APIResponse.error(
                "; ".join(result.errors),
                status_code=422,
            )

        response_data = {
            "success": True,
            "robot_name": result.robot_name,
            "links": len(result.links),
            "joints": len(result.joints),
            "warnings": result.warnings,
            "urdf": result.urdf_string,
        }

        if request.query_params.get("download") == "true":
            return APIResponse.file(result.urdf_string, f"{result.robot_name}.urdf")

        return APIResponse.ok(response_data)

    def convert_mjcf_to_urdf(self, request: APIRequest) -> APIResponse:
        """Convert MJCF to URDF."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.converters.mjcf_converter import (
            MJCFConverter,
        )

        body = request.body or {}

        content, error, _source = self._extract_text_content(
            request, body, "Missing MJCF content"
        )
        if error:
            return error

        converter = MJCFConverter()

        try:
            urdf_string = converter.mjcf_to_urdf(content)
        except (ValueError, KeyError, OSError) as e:
            return APIResponse.error(f"Conversion failed: {e}", 422)

        robot_name = body.get("robot_name", "converted")

        if request.query_params.get("download") == "true":
            return APIResponse.file(urdf_string, f"{robot_name}.urdf")

        return APIResponse.ok({"urdf": urdf_string})

    def convert_urdf_to_mjcf(self, request: APIRequest) -> APIResponse:
        """Convert URDF to MJCF."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.converters.mjcf_converter import (
            MJCFConverter,
        )

        body = request.body or {}

        content, error, _source = self._extract_text_content(
            request, body, "Missing URDF content"
        )
        if error:
            return error

        converter = MJCFConverter()

        try:
            mjcf_string = converter.urdf_to_mjcf(content)
        except (ValueError, KeyError, OSError) as e:
            return APIResponse.error(f"Conversion failed: {e}", 422)

        robot_name = body.get("robot_name", "converted")

        if request.query_params.get("download") == "true":
            return APIResponse.file(mjcf_string, f"{robot_name}.xml", "application/xml")

        return APIResponse.ok({"mjcf": mjcf_string})

    # ============================================================
    # Validation Handler
    # ============================================================

    def validate_urdf(self, request: APIRequest) -> APIResponse:
        """Validate URDF content."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.editor.text_editor import (
            URDFTextEditor,
            ValidationSeverity,
        )

        body = request.body or {}

        content, error, _source = self._extract_text_content(
            request, body, "Missing URDF content"
        )
        if error:
            return error

        editor = URDFTextEditor()
        editor.load_string(content)

        messages = editor.validate()

        has_errors = any(m.severity == ValidationSeverity.ERROR for m in messages)

        return APIResponse.ok(
            {
                "valid": not has_errors,
                "error_count": sum(
                    1 for m in messages if m.severity == ValidationSeverity.ERROR
                ),
                "warning_count": sum(
                    1 for m in messages if m.severity == ValidationSeverity.WARNING
                ),
                "messages": [
                    {
                        "severity": m.severity.value,
                        "line": m.line,
                        "column": m.column,
                        "message": m.message,
                        "element": m.element,
                    }
                    for m in messages
                ],
            }
        )

    # ============================================================
    # Parse Handler
    # ============================================================

    def parse_urdf(self, request: APIRequest) -> APIResponse:
        """Parse URDF and return structure."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.converters.urdf_parser import URDFParser

        body = request.body or {}

        content, error, _source = self._extract_text_content(
            request, body, "Missing URDF content"
        )
        if error:
            return error

        parser = URDFParser()

        try:
            model = parser.parse(content)
        except (ValueError, KeyError, OSError) as e:
            return APIResponse.error(f"Parse failed: {e}", 422)

        root = model.get_root_link()

        return APIResponse.ok(
            {
                "name": model.name,
                "root_link": root.name if root else None,
                "links": [link.to_dict() for link in model.links],
                "joints": [j.to_dict() for j in model.joints],
                "materials": {k: v.to_dict() for k, v in model.materials.items()},
                "warnings": model.warnings,
            }
        )

    # ============================================================
    # Inertia Handlers
    # ============================================================

    def calculate_inertia(self, request: APIRequest) -> APIResponse:
        """Calculate inertia for primitive shape."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.core.types import Inertia

        body = request.body or {}

        shape = body.get("shape")
        mass = body.get("mass", 1.0)
        dimensions = body.get("dimensions", [])

        if not shape:
            return APIResponse.error("Missing 'shape' parameter")

        try:
            if shape == "box":
                if len(dimensions) != 3:
                    return APIResponse.error("Box requires 3 dimensions")
                inertia = Inertia.from_box(mass, *dimensions)

            elif shape == "cylinder":
                if len(dimensions) != 2:
                    return APIResponse.error(
                        "Cylinder requires 2 dimensions (radius, length)"
                    )
                inertia = Inertia.from_cylinder(mass, dimensions[0], dimensions[1])

            elif shape == "sphere":
                if len(dimensions) != 1:
                    return APIResponse.error("Sphere requires 1 dimension (radius)")
                inertia = Inertia.from_sphere(mass, dimensions[0])

            elif shape == "capsule":
                if len(dimensions) != 2:
                    return APIResponse.error(
                        "Capsule requires 2 dimensions (radius, length)"
                    )
                inertia = Inertia.from_capsule(mass, dimensions[0], dimensions[1])

            else:
                return APIResponse.error(f"Unknown shape: {shape}")

        except (KeyError, ValueError, TypeError) as e:
            return APIResponse.error(f"Calculation failed: {e}")

        return APIResponse.ok(
            {
                "shape": shape,
                "mass": mass,
                "dimensions": dimensions,
                "inertia": {
                    "ixx": inertia.ixx,
                    "iyy": inertia.iyy,
                    "izz": inertia.izz,
                    "ixy": inertia.ixy,
                    "ixz": inertia.ixz,
                    "iyz": inertia.iyz,
                },
                "is_positive_definite": inertia.is_positive_definite(),
                "satisfies_triangle_inequality": inertia.satisfies_triangle_inequality(),
            }
        )

    def inertia_from_mesh(self, request: APIRequest) -> APIResponse:
        """Calculate inertia from mesh file."""
        if request is None:
            raise ValueError("request must be provided")
        body = request.body or {}

        mesh_content = request.files.get("mesh")
        if not mesh_content:
            return APIResponse.error("Missing mesh file")

        mass = body.get("mass")
        density = body.get("density")

        if mass is None and density is None:
            return APIResponse.error("Must provide either 'mass' or 'density'")
        try:
            mass_value = (
                self._require_positive_number(mass, "mass")
                if mass is not None
                else None
            )
            density_value = (
                self._require_positive_number(density, "density")
                if density is not None
                else None
            )
        except ValueError as e:
            return APIResponse.error(str(e))

        try:
            self._validate_mesh_upload(
                mesh_content,
                filename=str(body.get("filename") or ""),
            )
        except ValueError as e:
            return APIResponse.error(str(e), 413)

        # Derive the correct file suffix from the filename so trimesh uses the
        # right loader (OBJ, PLY, GLB, etc.).  Fall back to .stl for unknown types.
        _filename = str(body.get("filename") or "")
        _suffix = Path(_filename).suffix.lower() if _filename else ".stl"
        if _suffix not in ALLOWED_MESH_SUFFIXES:
            _suffix = ".stl"

        # Try to use trimesh for mesh-based inertia
        temp_path = ""
        try:
            import trimesh

            # Save to temp file; delete=False so trimesh can re-open it by path.
            # The finally block below guarantees cleanup on every exit path.
            with tempfile.NamedTemporaryFile(suffix=_suffix, delete=False) as f:
                f.write(mesh_content)
                temp_path = f.name

            mesh: Any = trimesh.load(temp_path)

            mesh_mass = getattr(mesh, "mass", None)
            try:
                mesh_mass = self._require_positive_number(mesh_mass, "mesh mass")
            except ValueError as e:
                return APIResponse.error(str(e))

            volume = mesh.volume
            if density_value is not None:
                mesh.density = density_value
                inertia_tensor = mesh.moment_inertia
                calculated_mass = mesh.mass
            else:
                if mass_value is None:
                    return APIResponse.error("Must provide either 'mass' or 'density'")
                # Scale inertia to specified mass
                inertia_tensor = mesh.moment_inertia * (mass_value / mesh_mass)
                calculated_mass = mass_value

            return APIResponse.ok(
                {
                    "mass": calculated_mass,
                    "volume": volume,
                    "center_of_mass": mesh.center_mass.tolist(),
                    "inertia": {
                        "ixx": float(inertia_tensor[0, 0]),
                        "iyy": float(inertia_tensor[1, 1]),
                        "izz": float(inertia_tensor[2, 2]),
                        "ixy": float(inertia_tensor[0, 1]),
                        "ixz": float(inertia_tensor[0, 2]),
                        "iyz": float(inertia_tensor[1, 2]),
                    },
                }
            )

        except ImportError:
            return APIResponse.error(
                "trimesh library not available for mesh-based inertia calculation",
                501,
            )
        except (PermissionError, OSError, ValueError, TypeError) as e:
            return APIResponse.error(f"Mesh processing failed: {e}")
        except Exception:
            logger.warning("Mesh parser failed", exc_info=True)
            return APIResponse.error("Mesh processing failed")
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    @staticmethod
    def _validate_mesh_upload(mesh_content: bytes, filename: str | None = None) -> None:
        """Validate mesh upload metadata and size before parser handoff."""
        if not mesh_content:
            raise ValueError("Mesh file is empty")
        if len(mesh_content) > MAX_MESH_UPLOAD_BYTES:
            raise ValueError(
                f"Mesh file exceeds {MAX_MESH_UPLOAD_BYTES // (1024 * 1024)} MiB limit"
            )
        if filename:
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_MESH_SUFFIXES:
                raise ValueError(f"Unsupported mesh file type: {suffix}")

    # ============================================================
    # Library Handlers
    # ============================================================

    def library_list_models(self, request: APIRequest) -> APIResponse:
        """List models in library."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.library import ModelLibrary

        library = ModelLibrary()

        category = request.query_params.get("category")
        source = request.query_params.get("source")
        search = request.query_params.get("search")
        tags = (
            request.query_params.get("tags", "").split(",")
            if request.query_params.get("tags")
            else None
        )

        models = library.list_models(
            category=category,
            source=source,
            search=search,
            tags=tags,
        )

        return APIResponse.ok(
            {
                "count": len(models),
                "models": [
                    {
                        "id": m.id,
                        "name": m.name,
                        "category": m.category.value,
                        "source": m.source.value if m.source else None,
                        "tags": m.tags,
                        "description": m.description,
                    }
                    for m in models
                ],
            }
        )

    def library_get_model(self, request: APIRequest) -> APIResponse:
        """Get model details."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.library import ModelLibrary

        model_id = request.query_params.get("model_id")
        if not model_id:
            return APIResponse.error("Missing model_id")

        library = ModelLibrary()
        models = library.list_models()

        for m in models:
            if m.id == model_id:
                return APIResponse.ok(
                    {
                        "id": m.id,
                        "name": m.name,
                        "category": m.category.value,
                        "source": m.source.value if m.source else None,
                        "tags": m.tags,
                        "description": m.description,
                        "path": str(m.urdf_path) if m.urdf_path else None,
                    }
                )

        return APIResponse.not_found(f"Model not found: {model_id}")

    def library_add_model(self, request: APIRequest) -> APIResponse:
        """Add model to library."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.library import ModelCategory, ModelLibrary

        body = request.body or {}

        content, error, _source = self._extract_text_content(
            request, body, "Missing URDF content"
        )
        if error:
            return error
        if content is None:
            return APIResponse.error("Missing URDF content")

        name = body.get("name", "unnamed")
        category_str = body.get("category", "other")
        tags = body.get("tags", [])

        # Parse category
        try:
            category = ModelCategory(category_str)
        except ValueError:
            category = ModelCategory.OTHER

        # Save to temp file and add
        library = ModelLibrary()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            entry = library.add_local_model(
                urdf_path=Path(temp_path),
                name=name,
                category=category,
                tags=tags,
            )

            if entry:
                return APIResponse.created(
                    {
                        "id": entry.id,
                        "name": entry.name,
                        "category": entry.category.value,
                    }
                )
            return APIResponse.error("Failed to add model")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def library_remove_model(self, request: APIRequest) -> APIResponse:
        """Remove model from library."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.library import ModelLibrary

        model_id = request.query_params.get("model_id")
        if not model_id:
            return APIResponse.error("Missing model_id")

        # ``delete_files`` is an opt-in flag (default False) mirroring
        # ModelLibrary.remove_model; accept the usual truthy query spellings.
        delete_files = str(request.query_params.get("delete_files", "")).lower() in (
            "1",
            "true",
            "yes",
        )

        library = ModelLibrary()
        removed = library.remove_model(model_id, delete_files=delete_files)
        if not removed:
            return APIResponse.not_found(f"Model not found: {model_id}")

        return APIResponse.ok({"removed": True, "id": model_id})

    def library_download_model(self, request: APIRequest) -> APIResponse:
        """Download model URDF."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.library import ModelLibrary

        model_id = request.query_params.get("model_id")
        if not model_id:
            return APIResponse.error("Missing model_id")

        library = ModelLibrary()
        model = library.load_model(model_id)

        if not model:
            return APIResponse.not_found(f"Model not found: {model_id}")

        urdf_string = model.to_urdf()

        return APIResponse.file(urdf_string, f"{model.name}.urdf")

    # ============================================================
    # Editor Handlers
    # ============================================================

    def compose_models(self, request: APIRequest) -> APIResponse:
        """Compose model from multiple sources."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.editor import FrankensteinEditor

        body = request.body or {}

        sources = body.get("sources", {})
        operations = body.get("operations", [])
        output_name = body.get("name", "composed_robot")

        if not sources:
            return APIResponse.error("Missing 'sources' in request body")

        editor = FrankensteinEditor()

        # Load source models
        for model_id, content in sources.items():
            try:
                editor.load_model(model_id, content, read_only=True)
            except (ValueError, KeyError, OSError) as e:
                return APIResponse.error(f"Failed to load model '{model_id}': {e}")

        # Create output model
        editor.create_model("output", output_name)

        # Process operations
        for op in operations:
            if not isinstance(op, dict):
                return APIResponse.error("Each operation must be an object")
            op_type = op.get("type")

            if op_type == "copy_subtree":
                error = self._validate_operation_fields(op, op_type, ("source", "link"))
                if error:
                    return error
                editor.copy_subtree(op["source"], op["link"])
            elif op_type == "paste":
                editor.paste(
                    "output",
                    attach_to=op.get("attach_to"),
                    prefix=op.get("prefix", ""),
                )
            elif op_type == "delete_subtree":
                error = self._validate_operation_fields(op, op_type, ("link",))
                if error:
                    return error
                editor.delete_subtree("output", op["link"])
            elif op_type == "rename":
                error = self._validate_operation_fields(
                    op, op_type, ("old_name", "new_name")
                )
                if error:
                    return error
                editor.rename_link("output", op["old_name"], op["new_name"])
            else:
                return APIResponse.error(f"Unknown operation type: {op_type}")

        # Export
        urdf_string = editor.export_model("output")
        stats = editor.get_model_statistics("output")

        if request.query_params.get("download") == "true":
            return APIResponse.file(urdf_string, f"{output_name}.urdf")

        return APIResponse.ok(
            {
                "name": output_name,
                "links": stats.get("link_count", 0),
                "joints": stats.get("joint_count", 0),
                "urdf": urdf_string,
            }
        )

    def diff_urdfs(self, request: APIRequest) -> APIResponse:
        """Compare two URDF files."""
        if request is None:
            raise ValueError("request must be provided")
        from shared.python.model_generation.editor.text_editor import URDFTextEditor

        body = request.body or {}

        content_a = body.get("content_a")
        content_b = body.get("content_b")

        if not content_a or not content_b:
            return APIResponse.error("Missing content_a or content_b")

        editor = URDFTextEditor()
        editor.load_string(content_a)

        diff_result = editor.get_diff_with_string(content_b)

        return APIResponse.ok(
            {
                "has_changes": diff_result.has_changes,
                "additions": diff_result.additions,
                "deletions": diff_result.deletions,
                "hunks": len(diff_result.hunks),
                "unified_diff": diff_result.unified_diff,
            }
        )
