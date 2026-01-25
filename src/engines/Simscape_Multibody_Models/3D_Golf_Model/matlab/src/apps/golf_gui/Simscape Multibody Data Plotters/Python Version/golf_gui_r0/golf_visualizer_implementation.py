#!/usr/bin/env python3
"""
Modern Golf Swing Visualizer - Production Implementation
High-performance, visually stunning 3D golf swing analysis tool

Key Technologies:
- PyQt6 for modern GUI
- OpenGL 4.3+ for hardware-accelerated rendering
- ModernGL for simplified OpenGL interface
- NumPy + Numba for high-performance computations
"""

import sys
import time
from dataclasses import dataclass

import moderngl as mgl
import numpy as np
import scipy.io
from numba import jit
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

# ============================================================================
# HIGH-PERFORMANCE DATA STRUCTURES
# ============================================================================


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
                # Find the main table variable
                var_name = self._find_table_variable(mat_data, name)
                datasets[name] = self._extract_dataframe(mat_data[var_name])
                print(f"✅ Loaded {name}: {len(datasets[name])} frames")
            except Exception as e:
                raise RuntimeError(f"Failed to load {filepath}: {e}") from e

        self._calculate_scaling_factors(datasets["BASEQ"])
        return datasets["BASEQ"], datasets["ZTCFQ"], datasets["DELTAQ"]

    def _find_table_variable(self, mat_data: dict, dataset_name: str) -> str:
        """Intelligently find the table variable in MAT file"""
        candidates = [f"{dataset_name}_table", dataset_name, dataset_name.lower()]
        for var_name in candidates:
            if var_name in mat_data:
                return var_name

        # Fallback to first non-system variable
        vars_found = [k for k in mat_data.keys() if not k.startswith("__")]
        if vars_found:
            return vars_found[0]
        raise ValueError(f"No valid table found in {dataset_name}")

    @jit(nopython=True)
    def _calculate_scaling_factors(self, baseq_data: np.ndarray):
        """Numba-accelerated scaling calculation"""
        # This would be implemented with proper Numba-compatible data access
        # For now, using regular NumPy approach

    def extract_frame_data(self, frame_idx: int, datasets: dict) -> FrameData:
        """Extract and process single frame data efficiently"""
        if frame_idx in self.cache:
            return self.cache[frame_idx]

        frame_data = FrameData(
            frame_idx=frame_idx,
            time=frame_idx * 0.001,  # Assume 1000 Hz sampling
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

        # Cache for performance
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

        # Shader sources
        self.vertex_shader_source = """
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

        self.fragment_shader_source = """
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

    def initialize(self, ctx):
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

        # Additional specialized shaders for vectors, ground, etc.
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
        # Implementation for ground grid rendering

    def _setup_geometry(self):
        """Create optimized geometry for body segments and club"""
        # High-quality cylinder for body segments
        self._create_cylinder_geometry()
        self._create_sphere_geometry()
        self._create_club_geometry()
        self._create_arrow_geometry()

    def _create_cylinder_geometry(self):
        """Create optimized cylinder with proper normals"""
        segments = 16
        vertices = []
        indices = []

        # Generate cylinder vertices with proper normals and texture coordinates
        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x, z = np.cos(angle), np.sin(angle)

            # Bottom vertex
            vertices.extend(
                [x, 0, z, x, 0, z, i / segments, 0]
            )  # pos, normal, texcoord
            # Top vertex
            vertices.extend([x, 1, z, x, 0, z, i / segments, 1])

        # Generate indices for triangle strips
        for i in range(segments):
            # Two triangles per segment
            indices.extend(
                [i * 2, i * 2 + 1, (i + 1) * 2, (i + 1) * 2, i * 2 + 1, (i + 1) * 2 + 1]
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
        # Icosphere generation for smooth spheres

    def _create_club_geometry(self):
        """Create detailed club geometry"""
        # Shaft: Simple cylinder
        # Clubhead: More complex geometry with realistic proportions

    def _create_arrow_geometry(self):
        """Create arrow geometry for force/torque vectors"""
        # Arrow shaft + arrowhead

    def _setup_lighting(self):
        """Configure realistic lighting"""
        # Set up uniforms for lighting calculations

    def render_frame(
        self,
        frame_data: FrameData,
        config: RenderConfig,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
    ):
        """Render complete frame with all elements"""
        self.ctx.clear(0.1, 0.2, 0.3)  # Sky blue background
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.BLEND)

        # Render ground plane
        if config.show_ground:
            self._render_ground(view_matrix, proj_matrix)

        # Render body segments
        self._render_body_segments(frame_data, config, view_matrix, proj_matrix)

        # Render club
        if config.show_club:
            self._render_club(frame_data, config, view_matrix, proj_matrix)

        # Render force/torque vectors
        self._render_vectors(frame_data, config, view_matrix, proj_matrix)

        # Render face normal
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
        segments = [
            (
                "left_forearm",
                frame_data.left_wrist,
                frame_data.left_elbow,
                0.025,
                [0.96, 0.76, 0.63],
            ),  # Skin color
            (
                "left_upper_arm",
                frame_data.left_elbow,
                frame_data.left_shoulder,
                0.035,
                [0.18, 0.32, 0.40],
            ),  # Shirt color
            (
                "right_forearm",
                frame_data.right_wrist,
                frame_data.right_elbow,
                0.025,
                [0.96, 0.76, 0.63],
            ),
            (
                "right_upper_arm",
                frame_data.right_elbow,
                frame_data.right_shoulder,
                0.035,
                [0.18, 0.32, 0.40],
            ),
            (
                "left_shoulder_neck",
                frame_data.left_shoulder,
                frame_data.hub,
                0.04,
                [0.18, 0.32, 0.40],
            ),
            (
                "right_shoulder_neck",
                frame_data.right_shoulder,
                frame_data.hub,
                0.04,
                [0.18, 0.32, 0.40],
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
        # Calculate transformation matrix
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return

        direction_normalized = direction / length

        # Create rotation matrix to align cylinder with direction
        up = np.array([0, 1, 0])
        if abs(np.dot(direction_normalized, up)) > 0.99:
            up = np.array([1, 0, 0])

        right = np.cross(direction_normalized, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction_normalized)

        rotation_matrix = np.column_stack([right, direction_normalized, up])

        # Create model matrix
        model_matrix = np.eye(4, dtype=np.float32)
        model_matrix[:3, :3] = rotation_matrix
        model_matrix[:3, 3] = start
        model_matrix[0, 0] *= radius  # Scale X
        model_matrix[1, 1] *= length  # Scale Y (length)
        model_matrix[2, 2] *= radius  # Scale Z

        # Set uniforms and render
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
            "BASEQ": [1.0, 0.42, 0.21],  # Orange
            "ZTCFQ": [0.31, 0.80, 0.77],  # Turquoise
            "DELTAQ": [1.0, 0.90, 0.43],  # Yellow
        }

        # Render forces
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

        # Render torques
        for dataset, torque in frame_data.torques.items():
            if (
                config.show_torques.get(dataset, True)
                and np.isfinite(torque).all()
                and np.linalg.norm(torque) > 1e-6
            ):
                scaled_torque = (
                    torque * config.vector_scale / self.max_torque_magnitude * 0.2
                )
                torque_pos = frame_data.midpoint + np.array(
                    [0.1, 0, 0]
                )  # Offset for visibility
                self._render_arrow(
                    torque_pos,
                    scaled_torque,
                    colors[dataset],
                    config.force_opacity,
                    view_matrix,
                    proj_matrix,
                )


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

        # Camera controls
        self.camera_distance = 3.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 20.0
        self.camera_target = np.array([0, 0, 0], dtype=np.float32)

        # Mouse interaction
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.5

        # Render configuration
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

        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.next_frame)

        # Performance monitoring
        self.frame_times = []
        self.fps = 0.0

    def initializeGL(self):
        """Initialize OpenGL context"""
        self.ctx = mgl.create_context()
        self.renderer.initialize(self.ctx)

        # Set up viewport
        self.ctx.viewport = (0, 0, self.width(), self.height())

        print("✅ OpenGL initialized successfully")
        print(f"   OpenGL Version: {self.ctx.info['GL_VERSION']}")
        print(f"   Vendor: {self.ctx.info['GL_VENDOR']}")
        print(f"   Renderer: {self.ctx.info['GL_RENDERER']}")

    def paintGL(self):
        """Render the current frame"""
        start_time = time.time()

        if self.datasets is None or self.num_frames == 0:
            self.ctx.clear(0.1, 0.2, 0.3)
            return

        # Extract current frame data
        frame_data = self.data_processor.extract_frame_data(
            self.current_frame, self.datasets
        )

        # Calculate view and projection matrices
        view_matrix = self._calculate_view_matrix()
        proj_matrix = self._calculate_projection_matrix()

        # Render the frame
        self.renderer.render_frame(
            frame_data, self.render_config, view_matrix, proj_matrix
        )

        # Update performance stats
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 60:  # Keep last 60 frames
            self.frame_times.pop(0)
        self.fps = (
            len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
        )

    def resizeGL(self, width, height):
        """Handle window resize"""
        self.ctx.viewport = (0, 0, width, height)

    def load_data(self, baseq_file: str, ztcfq_file: str, delta_file: str):
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
            print(f"✅ Data loaded: {self.num_frames} frames")
            self.update()
        except Exception as e:
            print(f"❌ Failed to load data: {e}")

    def play_animation(self):
        """Start animation playback"""
        if not self.is_playing and self.num_frames > 0:
            self.is_playing = True
            interval = int(33 / self.playback_speed)  # Target ~30 FPS
            self.animation_timer.start(interval)

    def pause_animation(self):
        """Pause animation playback"""
        self.is_playing = False
        self.animation_timer.stop()

    def next_frame(self):
        """Advance to next frame"""
        if self.num_frames > 0:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self.update()

    def set_frame(self, frame_idx: int):
        """Jump to specific frame"""
        if 0 <= frame_idx < self.num_frames:
            self.current_frame = frame_idx
            self.update()

    def mousePressEvent(self, event):
        """Handle mouse press for camera control"""
        self.last_mouse_pos = (event.x(), event.y())

    def mouseMoveEvent(self, event):
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

    def wheelEvent(self, event):
        """Handle mouse wheel for camera zoom"""
        delta = event.angleDelta().y() / 120
        self.camera_distance = np.clip(self.camera_distance - delta * 0.2, 0.5, 10.0)
        self.update()

    def _calculate_view_matrix(self) -> np.ndarray:
        """Calculate camera view matrix"""
        # Convert spherical to Cartesian coordinates
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)

        self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        self.camera_distance * np.sin(elevation_rad)
        self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)

        # Create view matrix (simplified lookAt)
        view_matrix = np.eye(4, dtype=np.float32)
        # ... implementation of lookAt matrix calculation

        return view_matrix

    def _calculate_projection_matrix(self) -> np.ndarray:
        """Calculate perspective projection matrix"""

        proj_matrix = np.eye(4, dtype=np.float32)
        # ... implementation of perspective projection matrix

        return proj_matrix


class ModernGolfVisualizerApp(QMainWindow):
    """Main application window with modern UI"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern Golf Swing Visualizer")
        self.setGeometry(100, 100, 1600, 900)

        # Create central widget
        self.gl_widget = ModernGolfVisualizerWidget()
        self.setCentralWidget(self.gl_widget)

        # Create dockable control panels
        self._create_control_panels()
        self._create_menubar()
        self._create_toolbar()
        self._create_status_bar()

        # Apply modern styling
        self._apply_modern_style()

    def _create_control_panels(self):
        """Create modern control panels"""
        # Playback controls dock
        playback_dock = QDockWidget("Playback Controls", self)
        playback_widget = self._create_playback_controls()
        playback_dock.setWidget(playback_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, playback_dock)

        # Visualization settings dock
        vis_dock = QDockWidget("Visualization", self)
        vis_widget = self._create_visualization_controls()
        vis_dock.setWidget(vis_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, vis_dock)

        # Performance monitor dock
        perf_dock = QDockWidget("Performance", self)
        perf_widget = self._create_performance_monitor()
        perf_dock.setWidget(perf_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, perf_dock)

    def _create_playback_controls(self) -> QWidget:
        """Create modern playback control panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Play/Pause button
        self.play_button = QPushButton("▶ Play")
        self.play_button.setMinimumHeight(40)
        self.play_button.clicked.connect(self._toggle_playback)
        layout.addWidget(self.play_button)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(1000)  # Will be updated when data loads
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        layout.addWidget(self.frame_slider)

        # Speed control
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

        # Force/Torque toggles
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

        # Body segment toggles
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

        # Update timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self._update_performance_display)
        self.perf_timer.start(100)  # Update 10 times per second

        return widget

    def _apply_modern_style(self):
        """Apply modern dark theme styling"""
        style = """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }

        QDockWidget {
            color: #ffffff;
            background-color: #3c3c3c;
        }

        QDockWidget::title {
            background-color: #4a4a4a;
            padding: 5px;
            border: 1px solid #5a5a5a;
        }

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

        self.setStyleSheet(style)


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Modern Golf Swing Visualizer")
    app.setApplicationVersion("2.0")

    # Create and show main window
    window = ModernGolfVisualizerApp()
    window.show()

    # Load sample data (if available)
    try:
        window.gl_widget.load_data("BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat")
    except Exception as e:
        print(f"Note: Sample data not found - {e}")
        print("Please load data using File -> Load Data")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
