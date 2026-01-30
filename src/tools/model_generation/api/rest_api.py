"""
REST API for model_generation package.

Provides HTTP endpoints for URDF generation, conversion, editing, and library access.
Can be used with Flask, FastAPI, or other frameworks via adapters.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HTTPMethod(Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIRequest:
    """Abstraction for HTTP request."""

    method: HTTPMethod
    path: str
    query_params: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | None = None
    files: dict[str, bytes] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class APIResponse:
    """Abstraction for HTTP response."""

    status_code: int
    body: dict[str, Any] | str | bytes
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: dict[str, Any]) -> APIResponse:
        return cls(status_code=200, body=data)

    @classmethod
    def created(cls, data: dict[str, Any]) -> APIResponse:
        return cls(status_code=201, body=data)

    @classmethod
    def error(cls, message: str, status_code: int = 400) -> APIResponse:
        return cls(status_code=status_code, body={"error": message})

    @classmethod
    def not_found(cls, message: str = "Not found") -> APIResponse:
        return cls(status_code=404, body={"error": message})

    @classmethod
    def file(
        cls,
        content: str | bytes,
        filename: str,
        content_type: str = "application/xml",
    ) -> APIResponse:
        return cls(
            status_code=200,
            body=content if isinstance(content, bytes) else content.encode(),
            content_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )


@dataclass
class Route:
    """API route definition."""

    method: HTTPMethod
    path: str
    handler: Callable[[APIRequest], APIResponse]
    description: str = ""
    tags: list[str] = field(default_factory=list)


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

    def __init__(self, prefix: str = "/api/v1"):
        """
        Initialize API.

        Args:
            prefix: URL prefix for all routes
        """
        self.prefix = prefix
        self._routes: list[Route] = []
        self._register_routes()

    def _register_routes(self) -> None:
        """Register all API routes."""
        # Health/info
        self.add_route(HTTPMethod.GET, "/health", self.health_check, "Health check")
        self.add_route(HTTPMethod.GET, "/info", self.get_api_info, "API information")

        # Generation endpoints
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

        # Conversion endpoints
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

        # Validation endpoint
        self.add_route(
            HTTPMethod.POST,
            "/validate",
            self.validate_urdf,
            "Validate URDF content",
            ["validation"],
        )

        # Parse endpoint
        self.add_route(
            HTTPMethod.POST,
            "/parse",
            self.parse_urdf,
            "Parse URDF and return structure",
            ["parsing"],
        )

        # Inertia calculation
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

        # Library endpoints
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

        # Editor endpoints
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

    def handle_request(self, request: APIRequest) -> APIResponse:
        """Handle an API request."""
        # Find matching route
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
                    return route.handler(request)
                except Exception as e:
                    logger.exception("Error handling request")
                    return APIResponse.error(str(e), 500)

        return APIResponse.not_found(
            f"No route for {request.method.value} {request.path}"
        )

    # ============================================================
    # Health/Info Handlers
    # ============================================================

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
        from model_generation.builders.parametric_builder import ParametricBuilder

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
        from model_generation.builders.manual_builder import ManualBuilder
        from model_generation.core.types import (
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
        from model_generation.converters.simscape import (
            ConversionConfig,
            SimscapeToURDFConverter,
        )

        body = request.body or {}

        # Get content from file upload or body
        content = None
        format_type = "mdl"

        if "file" in request.files:
            content = request.files["file"].decode("utf-8", errors="ignore")
            # Detect format from content
            if content.strip().startswith("<?xml") or content.strip().startswith("<"):
                format_type = "xml"
        elif "content" in body:
            content = body["content"]
            format_type = body.get("format", "mdl")
        else:
            return APIResponse.error("Missing model content or file")

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
        from model_generation.converters.mjcf_converter import MJCFConverter

        body = request.body or {}

        content = body.get("content") or (
            request.files.get("file", b"").decode("utf-8") if request.files else None
        )

        if not content:
            return APIResponse.error("Missing MJCF content")

        converter = MJCFConverter()

        try:
            urdf_string = converter.mjcf_to_urdf(content)
        except Exception as e:
            return APIResponse.error(f"Conversion failed: {e}", 422)

        robot_name = body.get("robot_name", "converted")

        if request.query_params.get("download") == "true":
            return APIResponse.file(urdf_string, f"{robot_name}.urdf")

        return APIResponse.ok({"urdf": urdf_string})

    def convert_urdf_to_mjcf(self, request: APIRequest) -> APIResponse:
        """Convert URDF to MJCF."""
        from model_generation.converters.mjcf_converter import MJCFConverter

        body = request.body or {}

        content = body.get("content") or (
            request.files.get("file", b"").decode("utf-8") if request.files else None
        )

        if not content:
            return APIResponse.error("Missing URDF content")

        converter = MJCFConverter()

        try:
            mjcf_string = converter.urdf_to_mjcf(content)
        except Exception as e:
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
        from model_generation.editor.text_editor import (
            URDFTextEditor,
            ValidationSeverity,
        )

        body = request.body or {}

        content = body.get("content") or (
            request.files.get("file", b"").decode("utf-8") if request.files else None
        )

        if not content:
            return APIResponse.error("Missing URDF content")

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
        from model_generation.converters.urdf_parser import URDFParser

        body = request.body or {}

        content = body.get("content") or (
            request.files.get("file", b"").decode("utf-8") if request.files else None
        )

        if not content:
            return APIResponse.error("Missing URDF content")

        parser = URDFParser()

        try:
            model = parser.parse(content)
        except Exception as e:
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
        from model_generation.core.types import Inertia

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

        except Exception as e:
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
        body = request.body or {}

        mesh_content = request.files.get("mesh")
        if not mesh_content:
            return APIResponse.error("Missing mesh file")

        mass = body.get("mass")
        density = body.get("density")

        if not mass and not density:
            return APIResponse.error("Must provide either 'mass' or 'density'")

        # Try to use trimesh for mesh-based inertia
        try:
            import trimesh

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
                f.write(mesh_content)
                temp_path = f.name

            mesh = trimesh.load(temp_path)

            if density:
                mesh.density = density
                inertia_tensor = mesh.moment_inertia
                calculated_mass = mesh.mass
            else:
                # Scale inertia to specified mass
                volume = mesh.volume
                inertia_tensor = mesh.moment_inertia * (mass / mesh.mass)
                calculated_mass = mass

            # Clean up
            Path(temp_path).unlink()

            return APIResponse.ok(
                {
                    "mass": calculated_mass,
                    "volume": volume if density else mesh.volume,
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
        except Exception as e:
            return APIResponse.error(f"Mesh processing failed: {e}")

    # ============================================================
    # Library Handlers
    # ============================================================

    def library_list_models(self, request: APIRequest) -> APIResponse:
        """List models in library."""
        from model_generation.library import ModelLibrary

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
        from model_generation.library import ModelLibrary

        model_id = request.query_params.get("model_id")
        if not model_id:
            return APIResponse.error("Missing model_id")

        library = ModelLibrary()
        models = library.list_models()

        for m in models:
            if m.model_id == model_id:
                return APIResponse.ok(
                    {
                        "id": m.model_id,
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
        from model_generation.library import ModelCategory, ModelLibrary

        body = request.body or {}

        content = body.get("content") or (
            request.files.get("file", b"").decode("utf-8") if request.files else None
        )

        if not content:
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
                        "id": entry.model_id,
                        "name": entry.name,
                        "category": entry.category.value,
                    }
                )
            else:
                return APIResponse.error("Failed to add model")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def library_remove_model(self, request: APIRequest) -> APIResponse:
        """Remove model from library."""
        from model_generation.library import ModelLibrary

        model_id = request.query_params.get("model_id")
        if not model_id:
            return APIResponse.error("Missing model_id")

        ModelLibrary()

        # Note: This would need implementation in ModelLibrary
        # For now, return not implemented
        return APIResponse.error("Remove not implemented", 501)

    def library_download_model(self, request: APIRequest) -> APIResponse:
        """Download model URDF."""
        from model_generation.library import ModelLibrary

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
        from model_generation.editor import FrankensteinEditor

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
            except Exception as e:
                return APIResponse.error(f"Failed to load model '{model_id}': {e}")

        # Create output model
        editor.create_model("output", output_name)

        # Process operations
        for op in operations:
            op_type = op.get("type")

            if op_type == "copy_subtree":
                editor.copy_subtree(op["source"], op["link"])
            elif op_type == "paste":
                editor.paste(
                    "output",
                    attach_to=op.get("attach_to"),
                    prefix=op.get("prefix", ""),
                )
            elif op_type == "delete_subtree":
                editor.delete_subtree("output", op["link"])
            elif op_type == "rename":
                editor.rename_link("output", op["old_name"], op["new_name"])

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
        from model_generation.editor.text_editor import URDFTextEditor

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


# ============================================================
# Framework Adapters
# ============================================================


class FlaskAdapter:
    """Adapter for Flask framework."""

    def __init__(self, api: ModelGenerationAPI):
        self.api = api

    def register(self, app: Any) -> None:
        """Register routes with Flask app."""
        from flask import jsonify, make_response
        from flask import request as flask_request

        for route in self.api.get_routes():
            endpoint = route.path.replace("/", "_").replace("{", "").replace("}", "")

            def make_handler(r: Route):
                def handler(**kwargs):
                    # Build APIRequest
                    api_request = APIRequest(
                        method=HTTPMethod(flask_request.method),
                        path=flask_request.path,
                        query_params={**flask_request.args, **kwargs},
                        body=flask_request.get_json(silent=True),
                        files={k: v.read() for k, v in flask_request.files.items()},
                        headers=dict(flask_request.headers),
                    )

                    response = self.api.handle_request(api_request)

                    if isinstance(response.body, bytes):
                        flask_response = make_response(response.body)
                    elif isinstance(response.body, dict):
                        flask_response = make_response(jsonify(response.body))
                    else:
                        flask_response = make_response(response.body)

                    flask_response.status_code = response.status_code
                    flask_response.content_type = response.content_type

                    for k, v in response.headers.items():
                        flask_response.headers[k] = v

                    return flask_response

                return handler

            # Convert path params from {param} to <param>
            flask_path = route.path.replace("{", "<").replace("}", ">")
            app.add_url_rule(
                flask_path,
                endpoint=endpoint,
                view_func=make_handler(route),
                methods=[route.method.value],
            )


class FastAPIAdapter:
    """Adapter for FastAPI framework."""

    def __init__(self, api: ModelGenerationAPI):
        self.api = api

    def register(self, app: Any) -> None:
        """Register routes with FastAPI app."""
        from fastapi import Request, Response
        from fastapi.responses import JSONResponse

        for route in self.api.get_routes():

            async def make_handler(r: Route):
                async def handler(request: Request, **kwargs):
                    body = None
                    try:
                        body = await request.json()
                    except Exception:
                        pass

                    files = {}
                    form = await request.form()
                    for key, value in form.items():
                        if hasattr(value, "read"):
                            files[key] = await value.read()

                    api_request = APIRequest(
                        method=HTTPMethod(request.method),
                        path=request.url.path,
                        query_params={**request.query_params, **kwargs},
                        body=body,
                        files=files,
                        headers=dict(request.headers),
                    )

                    response = self.api.handle_request(api_request)

                    if isinstance(response.body, bytes):
                        return Response(
                            content=response.body,
                            status_code=response.status_code,
                            media_type=response.content_type,
                            headers=response.headers,
                        )
                    else:
                        return JSONResponse(
                            content=response.body,
                            status_code=response.status_code,
                            headers=response.headers,
                        )

                return handler

            # FastAPI uses {param} format already
            app.add_api_route(
                route.path,
                make_handler(route),
                methods=[route.method.value],
                tags=route.tags,
                summary=route.description,
            )
