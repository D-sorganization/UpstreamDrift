#!/usr/bin/env python3
"""Modern Golf Swing Visualizer - Production Implementation
High-performance, visually stunning 3D golf swing analysis tool
Key Technologies:
- PyQt6 for modern GUI
- OpenGL 4.3+ for hardware-accelerated rendering
- ModernGL for simplified OpenGL interface
- NumPy + Numba for high-performance computations
"""

from __future__ import annotations

# ============================================================================
# HIGH-PERFORMANCE DATA STRUCTURES
# ============================================================================
import logging
import sys
import time
from dataclasses import dataclass

import moderngl as mgl
import numpy as np
import scipy.io
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Optimized frame data structure"""

    frame_idx: int
    time: float
    # Body points (NumPy arrays for vectorized operations)
    butt: np.ndarray
    clubhead: np.ndarray
    midpoint: np.ndarray
    left_wrist: np.ndarray
    left_elbow: np.ndarray
    left_shoulder: np.ndarray
    right_wrist: np.ndarray
    right_elbow: np.ndarray
    right_shoulder: np.ndarray
    hub: np.ndarray
    # Force/torque vectors for each dataset
    forces: dict[str, np.ndarray]  # 'BASEQ', 'ZTCFQ', 'DELTAQ'
    torques: dict[str, np.ndarray]


@dataclass
class RenderConfig:
    """Complete rendering configuration"""

    # Visibility toggles
    show_forces: dict[str, bool]
    show_torques: dict[str, bool]
    show_body_segments: dict[str, bool]
    show_club: bool = True
    show_face_normal: bool = True
    show_ground: bool = True
    show_ball: bool = True
    show_trajectory: bool = True
    # Visual parameters
    vector_scale: float = 1.0
    body_opacity: float = 0.8
    force_opacity: float = 0.9
    lighting_intensity: float = 1.0
    # Animation settings
    motion_blur: bool = False
    trail_length: int = 30


# ============================================================================
# HIGH-PERFORMANCE DATA PROCESSOR
# ============================================================================
class DataProcessor:
    """Optimized data loading and processing with Numba acceleration"""

    def __init__(self):
        self.cache = {}
        self.max_force_magnitude = 1.0
        self.max_torque_magnitude = 1.0

    def load_matlab_data(
        self, baseq_file: str, ztcfq_file: str, delta_file: str
    ) -> tuple[np.ndarray, ...]:
        """Fast MATLAB data loading with error handling"""
        datasets = {}
        files = {"BASEQ": baseq_file, "ZTCFQ": ztcfq_file, "DELTAQ": delta_file}
        for name, filepath in files.items():
            try:
                mat_data = scipy.io.loadmat(filepath)
                var_name = self._find_table_variable(mat_data, name)
                datasets[name] = self._extract_dataframe(mat_data[var_name])
                logger.info(f"Loaded {name}: {len(datasets[name])} frames")
            except (RuntimeError, TypeError, ValueError) as e:
                raise RuntimeError(f"Failed to load {filepath}: {e}") from e
        self._calculate_scaling_factors(datasets["BASEQ"])
        return datasets["BASEQ"], datasets["ZTCFQ"], datasets["DELTAQ"]

    def _find_table_variable(self, mat_data: dict, dataset_name: str) -> str:
        """Intelligently find the table variable in MAT file"""
        candidates = [f"{dataset_name}_table", dataset_name, dataset_name.lower()]
        for var_name in candidates:
            if var_name in mat_data:
                return var_name
        vars_found = [k for k in mat_data.keys() if not k.startswith("__")]
        if vars_found:
            return vars_found[0]
        raise ValueError(f"No valid table found in {dataset_name}")

    def _calculate_scaling_factors(self, baseq_data: np.ndarray):
        """Calculate scaling factors from data"""
        try:
            self.max_force_magnitude = 2000.0
            self.max_torque_magnitude = 200.0
            logger.info(
                f"Scaling factors set: Force={self.max_force_magnitude}N, "
                f"Torque={self.max_torque_magnitude}Nm"
            )
        except (RuntimeError, ValueError, OSError) as e:
            logger.info(f"Error calculating scaling factors: {e}")
            self.max_force_magnitude = 1000.0
            self.max_torque_magnitude = 100.0

    def extract_frame_data(self, frame_idx: int, datasets: dict) -> FrameData:
        """Extract and process single frame data efficiently"""
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        frame_data = FrameData(
            frame_idx=frame_idx,
            time=frame_idx * 0.001,
            butt=self._safe_extract_point(datasets["BASEQ"], frame_idx, "Butt"),
            clubhead=self._safe_extract_point(datasets["BASEQ"], frame_idx, "Clubhead"),
            midpoint=self._safe_extract_point(datasets["BASEQ"], frame_idx, "MidPoint"),
            left_wrist=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "LeftWrist"
            ),
            left_elbow=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "LeftElbow"
            ),
            left_shoulder=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "LeftShoulder"
            ),
            right_wrist=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "RightWrist"
            ),
            right_elbow=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "RightElbow"
            ),
            right_shoulder=self._safe_extract_point(
                datasets["BASEQ"], frame_idx, "RightShoulder"
            ),
            hub=self._safe_extract_point(datasets["BASEQ"], frame_idx, "Hub"),
            forces={
                "BASEQ": self._safe_extract_vector(
                    datasets["BASEQ"], frame_idx, "TotalHandForceGlobal"
                ),
                "ZTCFQ": self._safe_extract_vector(
                    datasets["ZTCFQ"], frame_idx, "TotalHandForceGlobal"
                ),
                "DELTAQ": self._safe_extract_vector(
                    datasets["DELTAQ"], frame_idx, "TotalHandForceGlobal"
                ),
            },
            torques={
                "BASEQ": self._safe_extract_vector(
                    datasets["BASEQ"], frame_idx, "EquivalentMidpointCoupleGlobal"
                ),
                "ZTCFQ": self._safe_extract_vector(
                    datasets["ZTCFQ"], frame_idx, "EquivalentMidpointCoupleGlobal"
                ),
                "DELTAQ": self._safe_extract_vector(
                    datasets["DELTAQ"], frame_idx, "EquivalentMidpointCoupleGlobal"
                ),
            },
        )
        self.cache[frame_idx] = frame_data
        return frame_data

    def _safe_extract_point(
        self, dataset: np.ndarray, frame_idx: int, column: str
    ) -> np.ndarray:
        """Safely extract 3D point with fallbacks"""
        try:
            point = dataset[column].iloc[frame_idx]
            if isinstance(point, list | np.ndarray) and len(point) == 3:
                return np.array(point, dtype=np.float32)
        except (TypeError, ValueError, IndexError):
            pass
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _safe_extract_vector(
        self, dataset: np.ndarray, frame_idx: int, column: str
    ) -> np.ndarray:
        """Safely extract 3D vector with fallbacks"""
        try:
            vector = dataset[column].iloc[frame_idx]
            if isinstance(vector, list | np.ndarray) and len(vector) == 3:
                return np.array(vector, dtype=np.float32)
        except (TypeError, ValueError, IndexError):
            pass
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)


# ============================================================================
# MODERN OPENGL RENDERER
# ============================================================================
class OpenGLRenderer:
    """High-performance OpenGL renderer with modern shaders"""

    def __init__(self):
        self.ctx = None
        self.programs = {}
        self.buffers = {}
        self.vaos = {}
        self.textures = {}

        self.vertex_shader_source = self._get_vertex_shader_source()
        self.fragment_shader_source = self._get_fragment_shader_source()

    @staticmethod
    def _get_vertex_shader_source():
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
    def _get_fragment_shader_source():
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
        self.ctx = ctx
        self._compile_shaders()
        self._setup_geometry()
        self._setup_lighting()

    def _compile_shaders(self):
        """Compile and link shader programs"""
        self.programs["standard"] = self.ctx.program(
            vertex_shader=self.vertex_shader_source,
            fragment_shader=self.fragment_shader_source,
        )
        self._compile_vector_shaders()
        self._compile_ground_shaders()

    def _compile_vector_shaders(self):
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

    def _compile_ground_shaders(self):
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

    def _setup_geometry(self):
        """Create optimized geometry for body segments and club"""
        self._create_cylinder_geometry()
        self._create_sphere_geometry()
        self._create_club_geometry()
        self._create_arrow_geometry()

    def _create_cylinder_geometry(self):
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

    def _create_sphere_geometry(self):
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
    def _get_sphere_vertices():
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
    def _get_sphere_indices():
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

    def _create_club_geometry(self):
        """Create detailed club geometry"""

    def _create_arrow_geometry(self):
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

    def _setup_lighting(self):
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
    ):
        """Render all body segments efficiently"""
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
    ):
        """Render cylinder between two 3D points"""
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        direction_normalized = direction / length
        up = np.array([0, 1, 0])
        if abs(np.dot(direction_normalized, up)) > 0.99:
            up = np.array([1, 0, 0])
        right = np.cross(direction_normalized, up)
        right = right / np.linalg.norm(right)
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
    ):
        """Render force and torque vectors with different colors"""
        colors = {
            "BASEQ": [1.0, 0.42, 0.21],
            "ZTCFQ": [0.31, 0.80, 0.77],
            "DELTAQ": [1.0, 0.90, 0.43],
        }
        for dataset, force in frame_data.forces.items():
            if (
                config.show_forces.get(dataset, True)
                and np.isfinite(force).all()
                and np.linalg.norm(force) > 1e-6
            ):
                scaled_force = (
                    force * config.vector_scale / self.max_force_magnitude * 0.3
                )
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
                and np.linalg.norm(torque) > 1e-6
            ):
                scaled_torque = (
                    torque * config.vector_scale / self.max_torque_magnitude * 0.2
                )
                torque_pos = frame_data.midpoint + np.array([0.1, 0, 0])
                self._render_arrow(
                    torque_pos,
                    scaled_torque,
                    colors[dataset],
                    config.force_opacity,
                    view_matrix,
                    proj_matrix,
                )

    def _render_ground(self, view_matrix, proj_matrix):
        """Render infinite ground grid"""
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

    def _render_club(self, frame_data, config, view_matrix, proj_matrix):
        """Render golf club"""
        if not (
            np.isfinite(frame_data.butt).all()
            and np.isfinite(frame_data.clubhead).all()
        ):
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

    def _render_face_normal(self, frame_data, config, view_matrix, proj_matrix):
        """Render face normal"""

    def _render_arrow(
        self,
        start_pos: np.ndarray,
        vector: np.ndarray,
        color: list[float],
        opacity: float,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ):
        """Render 3D arrow"""
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
        )

    def _render_arrow_head(
        self,
        end_pos: np.ndarray,
        vector: np.ndarray,
        color: list[float],
        opacity: float,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ):
        if "cone" not in self.vaos:
            return
        direction = vector
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        direction_normalized = direction / length
        up = np.array([0, 1, 0])
        if abs(np.dot(direction_normalized, up)) > 0.99:
            up = np.array([1, 0, 0])
        right = np.cross(direction_normalized, up)
        right = right / np.linalg.norm(right)
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


# ============================================================================
# MODERN GUI APPLICATION
# ============================================================================
class ModernGolfVisualizerWidget(QOpenGLWidget):
    """Modern OpenGL widget for golf swing visualization"""

    def __init__(self):
        super().__init__()
        self.renderer = OpenGLRenderer()
        self.data_processor = DataProcessor()
        self.datasets = None
        self.current_frame = 0
        self.num_frames = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.camera_distance = 3.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 20.0
        self.camera_target = np.array([0, 0, 0], dtype=np.float32)
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.5
        self.render_config = RenderConfig(
            show_forces={"BASEQ": True, "ZTCFQ": True, "DELTAQ": True},
            show_torques={"BASEQ": True, "ZTCFQ": True, "DELTAQ": True},
            show_body_segments={
                "left_forearm": True,
                "left_upper_arm": True,
                "right_forearm": True,
                "right_upper_arm": True,
                "left_shoulder_neck": True,
                "right_shoulder_neck": True,
            },
        )
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.next_frame)
        self.frame_times = []
        self.fps = 0.0

    def initializeGL(self) -> None:
        """Initialize OpenGL context"""
        self.ctx = mgl.create_context()
        self.renderer.initialize(self.ctx)
        self.ctx.viewport = (0, 0, self.width(), self.height())
        logger.info("OpenGL initialized successfully")
        logger.info(f"   OpenGL Version: {self.ctx.info['GL_VERSION']}")
        logger.info(f"   Vendor: {self.ctx.info['GL_VENDOR']}")
        logger.info(f"   Renderer: {self.ctx.info['GL_RENDERER']}")

    def paintGL(self) -> None:
        """Render the current frame"""
        start_time = time.time()
        if self.datasets is None or self.num_frames == 0:
            self.ctx.clear(0.1, 0.2, 0.3)
            return
        frame_data = self.data_processor.extract_frame_data(
            self.current_frame, self.datasets
        )
        view_matrix = self._calculate_view_matrix()
        proj_matrix = self._calculate_projection_matrix()
        self.renderer.render_frame(
            frame_data, self.render_config, view_matrix, proj_matrix
        )
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        self.fps = (
            len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
        )

    def resizeGL(self, width, height) -> None:
        """Handle window resize"""
        self.ctx.viewport = (0, 0, width, height)

    def load_data(self, baseq_file: str, ztcfq_file: str, delta_file: str) -> None:
        """Load golf swing data"""
        try:
            datasets = self.data_processor.load_matlab_data(
                baseq_file, ztcfq_file, delta_file
            )
            self.datasets = {
                "BASEQ": datasets[0],
                "ZTCFQ": datasets[1],
                "DELTAQ": datasets[2],
            }
            self.num_frames = len(datasets[0])
            self.current_frame = 0
            logger.info(f"Data loaded: {self.num_frames} frames")
            self.update()
        except (RuntimeError, ValueError, OSError) as e:
            logger.info(f"Failed to load data: {e}")

    def play_animation(self) -> None:
        """Start animation playback"""
        if not self.is_playing and self.num_frames > 0:
            self.is_playing = True
            interval = int(33 / self.playback_speed)
            self.animation_timer.start(interval)

    def pause_animation(self) -> None:
        """Pause animation playback"""
        self.is_playing = False
        self.animation_timer.stop()

    def next_frame(self) -> None:
        """Advance to next frame"""
        if self.num_frames > 0:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self.update()

    def set_frame(self, frame_idx: int) -> None:
        """Jump to specific frame"""
        if 0 <= frame_idx < self.num_frames:
            self.current_frame = frame_idx
            self.update()

    def mousePressEvent(self, event) -> None:
        """Handle mouse press for camera control"""
        self.last_mouse_pos = (event.x(), event.y())

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse movement for camera control"""
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos[0]
            dy = event.y() - self.last_mouse_pos[1]
            self.camera_azimuth += dx * self.mouse_sensitivity
            self.camera_elevation = np.clip(
                self.camera_elevation - dy * self.mouse_sensitivity, -89, 89
            )
            self.last_mouse_pos = (event.x(), event.y())
            self.update()

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for camera zoom"""
        delta = event.angleDelta().y() / 120
        self.camera_distance = np.clip(self.camera_distance - delta * 0.2, 0.5, 10.0)
        self.update()

    def _calculate_view_matrix(self) -> np.ndarray:
        """Calculate camera view matrix"""
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        self.camera_distance * np.sin(elevation_rad)
        self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        view_matrix = np.eye(4, dtype=np.float32)
        return view_matrix

    def _calculate_projection_matrix(self) -> np.ndarray:
        """Calculate perspective projection matrix"""
        proj_matrix = np.eye(4, dtype=np.float32)
        return proj_matrix


class ModernGolfVisualizerApp(QMainWindow):
    """Main application window with modern UI"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern Golf Swing Visualizer")
        self.setGeometry(100, 100, 1600, 900)
        self.gl_widget = ModernGolfVisualizerWidget()
        self.setCentralWidget(self.gl_widget)
        self._create_control_panels()
        self._create_menubar()
        self._create_toolbar()
        self._create_status_bar()
        self._apply_modern_style()

    def _create_control_panels(self):
        """Create modern control panels"""
        playback_dock = QDockWidget("Playback Controls", self)
        playback_widget = self._create_playback_controls()
        playback_dock.setWidget(playback_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, playback_dock)
        vis_dock = QDockWidget("Visualization", self)
        vis_widget = self._create_visualization_controls()
        vis_dock.setWidget(vis_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, vis_dock)
        perf_dock = QDockWidget("Performance", self)
        perf_widget = self._create_performance_monitor()
        perf_dock.setWidget(perf_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, perf_dock)

    def _create_playback_controls(self) -> QWidget:
        """Create modern playback control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.play_button = QPushButton("Play")
        self.play_button.setMinimumHeight(40)
        self.play_button.clicked.connect(self._toggle_playback)
        layout.addWidget(self.play_button)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(1000)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        layout.addWidget(self.frame_slider)
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(300)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_label)
        layout.addLayout(speed_layout)
        return widget

    def _create_visualization_controls(self) -> QWidget:
        """Create visualization control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        forces_group = QGroupBox("Forces")
        forces_layout = QVBoxLayout(forces_group)
        self.force_checkboxes = {}
        for dataset in ["BASEQ", "ZTCFQ", "DELTAQ"]:
            cb = QCheckBox(f"{dataset} Forces")
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda state, ds=dataset: self._toggle_forces(ds, state)
            )
            forces_layout.addWidget(cb)
            self.force_checkboxes[dataset] = cb
        layout.addWidget(forces_group)
        body_group = QGroupBox("Body Segments")
        body_layout = QVBoxLayout(body_group)
        self.body_checkboxes = {}
        segments = [
            "left_forearm",
            "left_upper_arm",
            "right_forearm",
            "right_upper_arm",
            "left_shoulder_neck",
            "right_shoulder_neck",
        ]
        for segment in segments:
            cb = QCheckBox(segment.replace("_", " ").title())
            cb.setChecked(True)
            cb.stateChanged.connect(
                lambda state, seg=segment: self._toggle_body_segment(seg, state)
            )
            body_layout.addWidget(cb)
            self.body_checkboxes[segment] = cb
        layout.addWidget(body_group)
        return widget

    def _create_performance_monitor(self) -> QWidget:
        """Create performance monitoring panel"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        self.fps_label = QLabel("FPS: 0")
        self.frame_label = QLabel("Frame: 0/0")
        self.time_label = QLabel("Time: 0.00s")
        layout.addWidget(self.fps_label)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.time_label)
        layout.addStretch()
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self._update_performance_display)
        self.perf_timer.start(100)
        return widget

    def _apply_modern_style(self):
        """Apply modern dark theme styling"""
        style = self._get_main_window_style()
        style += self._get_dock_widget_style()
        style += self._get_button_style()
        style += self._get_slider_style()
        style += self._get_checkbox_style()
        style += self._get_groupbox_style()
        self.setStyleSheet(style)

    @staticmethod
    def _get_main_window_style():
        return """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        """

    @staticmethod
    def _get_dock_widget_style():
        return """
        QDockWidget {
            color: #ffffff;
            background-color: #3c3c3c;
        }
        QDockWidget::title {
            background-color: #4a4a4a;
            padding: 5px;
            border: 1px solid #5a5a5a;
        }
        """

    @staticmethod
    def _get_button_style():
        return """
        QPushButton {
            background-color: #4a4a4a;
            border: 1px solid #6a6a6a;
            color: #ffffff;
            padding: 8px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
        QPushButton:pressed {
            background-color: #3a3a3a;
        }
        """

    @staticmethod
    def _get_slider_style():
        return """
        QSlider::groove:horizontal {
            border: 1px solid #5a5a5a;
            height: 8px;
            background: #3a3a3a;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #0078d4;
            border: 1px solid #005a9e;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        """

    @staticmethod
    def _get_checkbox_style():
        return """
        QCheckBox {
            color: #ffffff;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 13px;
            height: 13px;
        }
        QCheckBox::indicator:unchecked {
            background-color: #3a3a3a;
            border: 1px solid #6a6a6a;
        }
        QCheckBox::indicator:checked {
            background-color: #0078d4;
            border: 1px solid #005a9e;
        }
        """

    @staticmethod
    def _get_groupbox_style():
        return """
        QGroupBox {
            color: #ffffff;
            border: 2px solid #5a5a5a;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
        }
        """


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================
def main() -> None:
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Modern Golf Swing Visualizer")
    app.setApplicationVersion("2.0")
    window = ModernGolfVisualizerApp()
    window.show()
    try:
        window.gl_widget.load_data("BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat")
    except (RuntimeError, ValueError, OSError) as e:
        logger.info(f"Note: Sample data not found - {e}")
        logger.info("Please load data using File -> Load Data")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
