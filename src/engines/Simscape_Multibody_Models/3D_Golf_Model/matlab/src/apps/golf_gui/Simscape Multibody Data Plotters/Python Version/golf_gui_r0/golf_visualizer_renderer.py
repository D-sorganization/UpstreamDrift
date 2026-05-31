# mypy: disable-error-code="no-redef,var-annotated,attr-defined,assignment"
"""OpenGL renderer for the legacy golf swing visualizer."""

from __future__ import annotations

import logging
import math

import moderngl as mgl
import numpy as np

try:
    from .golf_visualizer_models import FrameData, RenderConfig
except ImportError:
    from golf_visualizer_models import FrameData, RenderConfig

logger = logging.getLogger(__name__)


class OpenGLRenderer:
    """High-performance OpenGL renderer with modern shaders"""

    def __init__(self) -> None:
        self.ctx = None
        self.programs = {}
        self.buffers = {}
        self.vaos = {}
        self.textures = {}

        self.vertex_shader_source = self._get_vertex_shader_source()
        self.fragment_shader_source = self._get_fragment_shader_source()

    @staticmethod
    def _get_vertex_shader_source() -> str:
        return """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec2 texCoord;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat3 normalMatrix;
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        void main() {
            FragPos = vec3(model * vec4(position, 1.0));
            Normal = normalMatrix * normal;
            TexCoord = texCoord;
            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """

    @staticmethod
    def _get_fragment_shader_source() -> str:
        return """
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        out vec4 FragColor;
        // Material properties
        uniform vec3 materialColor;
        uniform float materialSpecular;
        uniform float materialShininess;
        uniform float opacity;
        // Lighting
        uniform vec3 lightPosition;
        uniform vec3 lightColor;
        uniform vec3 viewPosition;
        uniform float ambientStrength;
        void main() {
            // Ambient lighting
            vec3 ambient = ambientStrength * lightColor;
            // Diffuse lighting
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPosition - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            // Specular lighting (Blinn-Phong)
            vec3 viewDir = normalize(viewPosition - FragPos);
            vec3 halfwayDir = normalize(lightDir + viewDir);
            float spec = pow(max(dot(norm, halfwayDir), 0.0), materialShininess);
            vec3 specular = materialSpecular * spec * lightColor;
            vec3 result = (ambient + diffuse + specular) * materialColor;
            FragColor = vec4(result, opacity);
        }
        """

    def initialize(self, ctx) -> None:
        """Initialize OpenGL context and resources"""
        if ctx is None:
            raise ValueError("ctx must be provided")
        self.ctx = ctx
        self._compile_shaders()
        self._setup_geometry()
        self._setup_lighting()

    def _compile_shaders(self) -> None:
        """Compile and link shader programs"""
        self.programs["standard"] = self.ctx.program(
            vertex_shader=self.vertex_shader_source,
            fragment_shader=self.fragment_shader_source,
        )
        self._compile_vector_shaders()
        self._compile_ground_shaders()

    def _compile_vector_shaders(self) -> None:
        """Compile shaders for force/torque vectors"""
        vector_vertex = """
        #version 330 core
        layout (location = 0) in vec3 position;
        uniform mat4 mvp;
        uniform vec3 start_pos;
        uniform vec3 vector;
        uniform float scale;
        void main() {
            vec3 world_pos = start_pos + position * scale * length(vector);
            gl_Position = mvp * vec4(world_pos, 1.0);
        }
        """
        vector_fragment = """
        #version 330 core
        out vec4 FragColor;
        uniform vec3 color;
        uniform float opacity;
        void main() {
            FragColor = vec4(color, opacity);
        }
        """
        self.programs["vector"] = self.ctx.program(
            vertex_shader=vector_vertex, fragment_shader=vector_fragment
        )

    def _compile_ground_shaders(self) -> None:
        """Compile shaders for ground plane with grid"""
        ground_vertex = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texCoord;
        uniform mat4 mvp;
        out vec2 uv;
        void main() {
            uv = texCoord;
            gl_Position = mvp * vec4(position, 1.0);
        }
        """
        ground_fragment = """
        #version 330 core
        in vec2 uv;
        out vec4 FragColor;
        uniform vec3 color;
        uniform float opacity;
        void main() {
            vec2 grid = abs(fract(uv * 20.0 - 0.5) - 0.5) / fwidth(uv * 20.0);
            float line = min(grid.x, grid.y);
            float alpha = 1.0 - min(line, 1.0);
            vec3 gridColor = vec3(0.8);
            vec3 groundColor = color * 0.3;
            vec3 finalColor = mix(groundColor, gridColor, alpha * 0.3);
            FragColor = vec4(finalColor, opacity);
        }
        """
        try:
            self.programs["ground"] = self.ctx.program(
                vertex_shader=ground_vertex, fragment_shader=ground_fragment
            )
        except (RuntimeError, ValueError, OSError) as e:
            logger.info(f"Failed to compile ground shader: {e}")

    def _setup_geometry(self) -> None:
        """Create optimized geometry for body segments and club"""
        self._create_cylinder_geometry()
        self._create_sphere_geometry()
        self._create_club_geometry()
        self._create_arrow_geometry()

    def _create_cylinder_geometry(self) -> None:
        """Create optimized cylinder with proper normals"""
        segments = 16
        vertices = []
        indices = []
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x, z = np.cos(angle), np.sin(angle)
            vertices.extend([x, 0, z, x, 0, z, i / segments, 0])
            vertices.extend([x, 1, z, x, 0, z, i / segments, 1])
        for i in range(segments):
            indices.extend(
                [
                    i * 2,
                    i * 2 + 1,
                    (i + 1) * 2,
                    (i + 1) * 2,
                    i * 2 + 1,
                    (i + 1) * 2 + 1,
                ]
            )
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        self.buffers["cylinder_vbo"] = self.ctx.buffer(vertices)
        self.buffers["cylinder_ebo"] = self.ctx.buffer(indices)
        self.vaos["cylinder"] = self.ctx.vertex_array(
            self.programs["standard"],
            [
                (
                    self.buffers["cylinder_vbo"],
                    "3f 3f 2f",
                    "position",
                    "normal",
                    "texCoord",
                )
            ],
            self.buffers["cylinder_ebo"],
        )

    def _create_sphere_geometry(self) -> None:
        """Create optimized sphere geometry"""
        vertices = self._get_sphere_vertices()
        indices = self._get_sphere_indices()
        self.buffers["sphere_vbo"] = self.ctx.buffer(vertices)
        self.buffers["sphere_ebo"] = self.ctx.buffer(indices)
        self.vaos["sphere"] = self.ctx.vertex_array(
            self.programs["standard"],
            [
                (
                    self.buffers["sphere_vbo"],
                    "3f 3f 2f",
                    "position",
                    "normal",
                    "texCoord",
                )
            ],
            self.buffers["sphere_ebo"],
        )

    @staticmethod
    def _get_sphere_vertices() -> np.ndarray:
        return np.array(
            [
                # fmt: off
                -0.5,
                -0.5,
                -0.5,
                0,
                0,
                -1,
                0,
                0,
                0.5,
                -0.5,
                -0.5,
                0,
                0,
                -1,
                1,
                0,
                0.5,
                0.5,
                -0.5,
                0,
                0,
                -1,
                1,
                1,
                -0.5,
                0.5,
                -0.5,
                0,
                0,
                -1,
                0,
                1,
                -0.5,
                -0.5,
                0.5,
                0,
                0,
                1,
                0,
                0,
                0.5,
                -0.5,
                0.5,
                0,
                0,
                1,
                1,
                0,
                0.5,
                0.5,
                0.5,
                0,
                0,
                1,
                1,
                1,
                -0.5,
                0.5,
                0.5,
                0,
                0,
                1,
                0,
                1,
                # fmt: on
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _get_sphere_indices() -> np.ndarray:
        return np.array(
            [
                0,
                1,
                2,
                2,
                3,
                0,
                4,
                5,
                6,
                6,
                7,
                4,
                0,
                4,
                7,
                7,
                3,
                0,
                1,
                5,
                6,
                6,
                2,
                1,
                0,
                1,
                5,
                5,
                4,
                0,
                3,
                2,
                6,
                6,
                7,
                3,
            ],
            dtype=np.uint32,
        )

    def _create_club_geometry(self) -> None:
        """Create detailed club geometry"""

    def _create_arrow_geometry(self) -> None:
        """Create arrow geometry for force/torque vectors"""
        segments = 16
        vertices = []
        indices = []
        vertices.extend([0, 1, 0, 0, 1, 0, 0.5, 1])
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x, z = np.cos(angle), np.sin(angle)
            vertices.extend([x, 0, z, x, 0.5, z, i / segments, 0])
        vertices = np.array(vertices, dtype=np.float32)
        for i in range(segments):
            indices.extend([0, i + 1, (i + 1) % segments + 1])
        indices = np.array(indices, dtype=np.uint32)
        self.buffers["cone_vbo"] = self.ctx.buffer(vertices)
        self.buffers["cone_ebo"] = self.ctx.buffer(indices)
        self.vaos["cone"] = self.ctx.vertex_array(
            self.programs["standard"],
            [
                (
                    self.buffers["cone_vbo"],
                    "3f 3f 2f",
                    "position",
                    "normal",
                    "texCoord",
                )
            ],
            self.buffers["cone_ebo"],
        )

    def _setup_lighting(self) -> None:
        """Configure realistic lighting"""
        if "standard" in self.programs:
            prog = self.programs["standard"]
            try:
                if "lightPosition" in prog:
                    prog["lightPosition"].value = (5.0, 10.0, 5.0)
                if "lightColor" in prog:
                    prog["lightColor"].value = (1.0, 1.0, 1.0)
                if "ambientStrength" in prog:
                    prog["ambientStrength"].value = 0.4
                if "materialSpecular" in prog:
                    prog["materialSpecular"].value = 0.5
                if "materialShininess" in prog:
                    prog["materialShininess"].value = 32.0
            except (RuntimeError, ValueError, OSError) as e:
                logger.info(f"Lighting setup warning: {e}")

    def render_frame(
        self,
        frame_data: FrameData,
        config: RenderConfig,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ) -> None:
        """Render complete frame with all elements"""
        if frame_data is None:
            raise ValueError("frame_data must be provided")
        self.ctx.clear(0.1, 0.2, 0.3)
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.BLEND)
        if config.show_ground:
            self._render_ground(view_matrix, proj_matrix)
        self._render_body_segments(frame_data, config, view_matrix, proj_matrix)
        if config.show_club:
            self._render_club(frame_data, config, view_matrix, proj_matrix)
        self._render_vectors(frame_data, config, view_matrix, proj_matrix)
        if config.show_face_normal:
            self._render_face_normal(frame_data, config, view_matrix, proj_matrix)

    def _render_body_segments(
        self,
        frame_data: FrameData,
        config: RenderConfig,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ) -> None:
        """Render all body segments efficiently"""
        if frame_data is None:
            raise ValueError("frame_data must be provided")
        skin = [0.96, 0.76, 0.63]
        dark = [0.18, 0.32, 0.40]
        segments = [
            ("left_forearm", frame_data.left_wrist, frame_data.left_elbow, 0.025, skin),
            (
                "left_upper_arm",
                frame_data.left_elbow,
                frame_data.left_shoulder,
                0.035,
                dark,
            ),
            (
                "right_forearm",
                frame_data.right_wrist,
                frame_data.right_elbow,
                0.025,
                skin,
            ),
            (
                "right_upper_arm",
                frame_data.right_elbow,
                frame_data.right_shoulder,
                0.035,
                dark,
            ),
            (
                "left_shoulder_neck",
                frame_data.left_shoulder,
                frame_data.hub,
                0.04,
                dark,
            ),
            (
                "right_shoulder_neck",
                frame_data.right_shoulder,
                frame_data.hub,
                0.04,
                dark,
            ),
        ]
        for segment_name, start_pos, end_pos, radius, color in segments:
            if not config.show_body_segments.get(segment_name, True):
                continue
            if not (np.isfinite(start_pos).all() and np.isfinite(end_pos).all()):
                continue
            self._render_cylinder_between_points(
                start_pos,
                end_pos,
                radius,
                color,
                config.body_opacity,
                view_matrix,
                proj_matrix,
            )

    def _render_cylinder_between_points(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: list[float],
        opacity: float,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ) -> None:
        """Render cylinder between two 3D points"""
        if start is None:
            raise ValueError("start must be provided")
        direction = end - start
        # ⚡ Bolt: math.sqrt(np.dot) is faster than np.linalg.norm for small 1D arrays
        length = math.sqrt(np.dot(direction, direction))
        if length < 1e-6:
            return
        direction_normalized = direction / length
        up = np.array([0, 1, 0])
        if abs(np.dot(direction_normalized, up)) > 0.99:
            up = np.array([1, 0, 0])
        right = np.cross(direction_normalized, up)
        # ⚡ Bolt: math.sqrt(np.dot) is faster than np.linalg.norm for small 1D arrays
        right = right / math.sqrt(np.dot(right, right))
        up = np.cross(right, direction_normalized)
        rotation_matrix = np.column_stack([right, direction_normalized, up])
        model_matrix = np.eye(4, dtype=np.float32)
        model_matrix[:3, :3] = rotation_matrix
        model_matrix[:3, 3] = start
        model_matrix[0, 0] *= radius
        model_matrix[1, 1] *= length
        model_matrix[2, 2] *= radius
        self.programs["standard"]["model"].write(model_matrix.tobytes())
        self.programs["standard"]["view"].write(view_matrix.tobytes())
        self.programs["standard"]["projection"].write(proj_matrix.tobytes())
        self.programs["standard"]["materialColor"].value = tuple(color)
        self.programs["standard"]["opacity"].value = opacity
        self.vaos["cylinder"].render()

    def _render_vectors(
        self,
        frame_data: FrameData,
        config: RenderConfig,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ) -> None:
        """Render force and torque vectors with different colors"""
        if frame_data is None:
            raise ValueError("frame_data must be provided")
        colors = {
            "BASEQ": [1.0, 0.42, 0.21],
            "ZTCFQ": [0.31, 0.80, 0.77],
            "DELTAQ": [1.0, 0.90, 0.43],
        }
        for dataset, force in frame_data.forces.items():
            if (
                config.show_forces.get(dataset, True)
                and np.isfinite(force).all()
                and math.sqrt(np.dot(force, force)) > 1e-6
            ):
                scaled_force = (
                    force * config.vector_scale / self.max_force_magnitude * 0.3
                )  # noqa: E501
                self._render_arrow(
                    frame_data.midpoint,
                    scaled_force,
                    colors[dataset],
                    config.force_opacity,
                    view_matrix,
                    proj_matrix,
                )
        for dataset, torque in frame_data.torques.items():
            if (
                config.show_torques.get(dataset, True)
                and np.isfinite(torque).all()
                and math.sqrt(np.dot(torque, torque)) > 1e-6
            ):
                scaled_torque = (
                    torque * config.vector_scale / self.max_torque_magnitude * 0.2
                )  # noqa: E501
                torque_pos = frame_data.midpoint + np.array([0.1, 0, 0])
                self._render_arrow(
                    torque_pos,
                    scaled_torque,
                    colors[dataset],
                    config.force_opacity,
                    view_matrix,
                    proj_matrix,
                )

    def _render_ground(self, view_matrix, proj_matrix) -> None:
        """Render infinite ground grid"""
        if view_matrix is None:
            raise ValueError("view_matrix must be provided")
        if "ground" not in self.vaos:
            size = 50.0
            vertices = np.array(
                [
                    -size,
                    0,
                    -size,
                    0,
                    0,
                    size,
                    0,
                    -size,
                    1,
                    0,
                    size,
                    0,
                    size,
                    1,
                    1,
                    -size,
                    0,
                    size,
                    0,
                    1,
                ],
                dtype=np.float32,
            )
            indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
            self.buffers["ground_vbo"] = self.ctx.buffer(vertices)
            self.buffers["ground_ebo"] = self.ctx.buffer(indices)
            if "ground" in self.programs:
                self.vaos["ground"] = self.ctx.vertex_array(
                    self.programs["ground"],
                    [(self.buffers["ground_vbo"], "3f 2f", "position", "texCoord")],
                    self.buffers["ground_ebo"],
                )
        if "ground" in self.vaos and "ground" in self.programs:
            mvp = proj_matrix @ view_matrix
            self.programs["ground"]["mvp"].write(mvp.tobytes())
            self.programs["ground"]["color"].value = (0.2, 0.6, 0.2)
            self.programs["ground"]["opacity"].value = 1.0
            self.vaos["ground"].render()

    def _render_club(self, frame_data, config, view_matrix, proj_matrix) -> None:
        """Render golf club"""
        if frame_data is None:
            raise ValueError("frame_data must be provided")
        if not (
            np.isfinite(frame_data.butt).all()
            and np.isfinite(frame_data.clubhead).all()
        ):  # noqa: E501
            return
        self._render_cylinder_between_points(
            frame_data.butt,
            frame_data.clubhead,
            0.015,
            [0.8, 0.8, 0.8],
            config.body_opacity,
            view_matrix,
            proj_matrix,
        )
        if "sphere" in self.vaos:
            model_matrix = np.eye(4, dtype=np.float32)
            model_matrix[:3, 3] = frame_data.clubhead
            s = 0.05
            model_matrix[0, 0] = s
            model_matrix[1, 1] = s
            model_matrix[2, 2] = s
            self.programs["standard"]["model"].write(model_matrix.tobytes())
            self.programs["standard"]["view"].write(view_matrix.tobytes())
            self.programs["standard"]["projection"].write(proj_matrix.tobytes())
            self.programs["standard"]["materialColor"].value = (0.2, 0.2, 0.2)
            self.programs["standard"]["opacity"].value = config.body_opacity
            self.vaos["sphere"].render()

    def _render_face_normal(self, frame_data, config, view_matrix, proj_matrix) -> None:
        """Render face normal"""

    def _render_arrow(
        self,
        start_pos: np.ndarray,
        vector: np.ndarray,
        color: list[float],
        opacity: float,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ) -> None:
        """Render 3D arrow"""
        if start_pos is None:
            raise ValueError("start_pos must be provided")
        end_pos = start_pos + vector
        self._render_cylinder_between_points(
            start_pos,
            end_pos,
            0.01,
            color,
            opacity,
            view_matrix,
            proj_matrix,
        )
        self._render_arrow_head(
            end_pos, vector, color, opacity, view_matrix, proj_matrix
        )  # noqa: E501

    def _render_arrow_head(
        self,
        end_pos: np.ndarray,
        vector: np.ndarray,
        color: list[float],
        opacity: float,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ) -> None:
        if end_pos is None:
            raise ValueError("end_pos must be provided")
        if "cone" not in self.vaos:
            return
        direction = vector
        # ⚡ Bolt: math.sqrt(np.dot) is faster than np.linalg.norm for small 1D arrays
        length = math.sqrt(np.dot(direction, direction))
        if length < 1e-6:
            return
        direction_normalized = direction / length
        up = np.array([0, 1, 0])
        if abs(np.dot(direction_normalized, up)) > 0.99:
            up = np.array([1, 0, 0])
        right = np.cross(direction_normalized, up)
        # ⚡ Bolt: math.sqrt(np.dot) is faster than np.linalg.norm for small 1D arrays
        right = right / math.sqrt(np.dot(right, right))
        up = np.cross(right, direction_normalized)
        rotation_matrix = np.column_stack([right, direction_normalized, up])
        model_matrix = np.eye(4, dtype=np.float32)
        model_matrix[:3, :3] = rotation_matrix
        model_matrix[:3, 3] = end_pos
        s = 0.04
        model_matrix[0, 0] = s
        model_matrix[1, 1] = s * 2.0
        model_matrix[2, 2] = s
        self.programs["standard"]["model"].write(model_matrix.tobytes())
        self.programs["standard"]["view"].write(view_matrix.tobytes())
        self.programs["standard"]["projection"].write(proj_matrix.tobytes())
        self.programs["standard"]["materialColor"].value = tuple(color)
        self.programs["standard"]["opacity"].value = opacity
        self.vaos["cone"].render()
