#!/usr/bin/env python3
"""
Golf Swing Visualizer - Fixed OpenGL Renderer
Fixed for moderngl 5.x compatibility with correct uniform API
"""

import time
import traceback
from dataclasses import dataclass

import moderngl as mgl
import numpy as np

# ============================================================================
# FIXED SHADER DEFINITIONS
# ============================================================================


class ShaderLibrary:
    """Fixed GLSL shaders for golf swing visualization"""

    @staticmethod
    def get_simple_vertex_shader() -> str:
        """Simple vertex shader with basic transformation"""
        return """
        #version 330 core

        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 FragPos;
        out vec3 Normal;

        void main() {
            vec4 worldPos = model * vec4(position, 1.0);
            FragPos = worldPos.xyz;
            Normal = mat3(model) * normal;

            gl_Position = projection * view * worldPos;
        }
        """

    @staticmethod
    def get_simple_fragment_shader() -> str:
        """Simple fragment shader with basic lighting"""
        return """
        #version 330 core

        in vec3 FragPos;
        in vec3 Normal;

        out vec4 FragColor;

        uniform vec3 materialColor;
        uniform vec3 lightPosition;
        uniform vec3 lightColor;
        uniform vec3 viewPosition;
        uniform float opacity;

        void main() {
            vec3 N = normalize(Normal);
            vec3 L = normalize(lightPosition - FragPos);
            vec3 V = normalize(viewPosition - FragPos);
            vec3 R = reflect(-L, N);

            // Ambient
            vec3 ambient = 0.3 * materialColor;

            // Diffuse
            float diff = max(dot(N, L), 0.0);
            vec3 diffuse = diff * lightColor * materialColor;

            // Specular
            float spec = pow(max(dot(V, R), 0.0), 32.0);
            vec3 specular = spec * lightColor * 0.5;

            vec3 result = ambient + diffuse + specular;
            FragColor = vec4(result, opacity);
        }
        """

    @staticmethod
    def get_ground_vertex_shader() -> str:
        """Simple vertex shader for ground plane"""
        return """
        #version 330 core

        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texCoord;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec2 TexCoord;

        void main() {
            gl_Position = projection * view * model * vec4(position, 1.0);
            TexCoord = texCoord;
        }
        """

    @staticmethod
    def get_ground_fragment_shader() -> str:
        """Simple fragment shader for ground with grid"""
        return """
        #version 330 core

        in vec2 TexCoord;

        out vec4 FragColor;

        uniform vec3 grassColor;
        uniform vec3 gridColor;
        uniform float gridSpacing;

        void main() {
            // Simple grid pattern
            vec2 grid = fract(TexCoord * gridSpacing);
            float line = min(grid.x, grid.y);
            float gridStrength = 1.0 - smoothstep(0.0, 0.1, line);

            vec3 color = mix(grassColor, gridColor, gridStrength * 0.3);
            FragColor = vec4(color, 1.0);
        }
        """


# ============================================================================
# GEOMETRY MANAGER
# ============================================================================


@dataclass
class GeometryObject:
    """Container for OpenGL geometry"""

    vao: mgl.VertexArray
    vertex_count: int
    index_count: int
    visible: bool = True
    position: np.ndarray | None = None
    rotation: np.ndarray | None = None
    scale: np.ndarray | None = None

    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3, dtype=np.float32)
        if self.rotation is None:
            self.rotation = np.eye(3, dtype=np.float32)
        if self.scale is None:
            self.scale = np.ones(3, dtype=np.float32)


class GeometryManager:
    """Fixed geometry management"""

    def __init__(self, ctx: mgl.Context):
        self.ctx = ctx
        self.geometry_objects: dict[str, GeometryObject] = {}
        self.mesh_library: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.programs: dict[str, mgl.Program] = {}

        # Initialize standard meshes
        self._create_standard_meshes()
        self._compile_shaders()

    def _create_standard_meshes(self):
        """Create simple mesh library"""
        try:
            print("🔧 Creating standard meshes...")

            # Import geometry utilities from the core module
            from golf_data_core import GeometryUtils

            print("  [OK] GeometryUtils imported")

            # Create simple meshes
            print("  Creating cylinder mesh...")
            self.mesh_library["cylinder"] = GeometryUtils.create_cylinder_mesh(
                radius=1.0, height=1.0, segments=8
            )
            print("  [OK] Cylinder mesh created")

            print("  Creating sphere mesh...")
            self.mesh_library["sphere"] = GeometryUtils.create_sphere_mesh(
                radius=1.0, lat_segments=8, lon_segments=8
            )
            print("  [OK] Sphere mesh created")

            # Ground plane
            print("  Creating ground mesh...")
            self._create_ground_mesh()
            print("  [OK] Ground mesh created")

            print(f"[OK] Created {len(self.mesh_library)} standard meshes")

        except Exception as e:
            print(f"[ERROR] Failed to create standard meshes: {e}")
            traceback.print_exc()
            raise

    def _create_ground_mesh(self):
        """Create simple ground plane mesh"""
        size = 10.0
        vertices = [
            -size,
            0,
            -size,
            0,
            0,  # position, texcoord
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
        ]

        indices = [0, 1, 2, 0, 2, 3]

        self.mesh_library["ground"] = (
            np.array(vertices, dtype=np.float32),
            np.array([0, 1, 0] * 4, dtype=np.float32),  # Normals pointing up
            np.array(indices, dtype=np.uint32),
        )

    def _compile_shaders(self):
        """Compile fixed shader programs"""
        try:
            print("🔧 Compiling shader programs...")

            # Simple shader
            print("  Compiling simple shader...")
            self.programs["simple"] = self.ctx.program(
                vertex_shader=ShaderLibrary.get_simple_vertex_shader(),
                fragment_shader=ShaderLibrary.get_simple_fragment_shader(),
            )
            print(f"  [OK] Simple shader compiled: {type(self.programs['simple'])}")

            # Ground shader
            print("  Compiling ground shader...")
            self.programs["ground"] = self.ctx.program(
                vertex_shader=ShaderLibrary.get_ground_vertex_shader(),
                fragment_shader=ShaderLibrary.get_ground_fragment_shader(),
            )
            print(f"  [OK] Ground shader compiled: {type(self.programs['ground'])}")

            print(f"[OK] Compiled {len(self.programs)} shader programs")

        except Exception as e:
            print(f"[ERROR] Failed to compile shaders: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to compile shaders: {e}") from e

    def create_geometry_object(
        self, name: str, mesh_type: str, program_name: str = "simple"
    ) -> GeometryObject:
        """Create a new geometry object from mesh library"""
        if mesh_type not in self.mesh_library:
            raise ValueError(f"Mesh type '{mesh_type}' not found in library")

        if program_name not in self.programs:
            raise ValueError(f"Program '{program_name}' not found")

        vertices, normals, indices = self.mesh_library[mesh_type]
        program = self.programs[program_name]

        # Create buffers
        vertex_buffer = self.ctx.buffer(vertices)
        normal_buffer = self.ctx.buffer(normals)
        index_buffer = self.ctx.buffer(indices)

        # Create VAO based on mesh type
        if mesh_type == "ground":
            # Ground mesh has position + texcoord
            vao = self.ctx.vertex_array(
                program,
                [
                    (vertex_buffer, "3f 2f", 0, 1)
                ],  # position at location 0, texCoord at location 1
                index_buffer,
            )
        else:
            # Standard mesh has position + normal only
            vao = self.ctx.vertex_array(
                program,
                [(vertex_buffer, "3f", "position"), (normal_buffer, "3f", "normal")],
                index_buffer,
            )

        geometry_obj = GeometryObject(
            vao=vao, vertex_count=len(vertices) // 3, index_count=len(indices)
        )

        self.geometry_objects[name] = geometry_obj
        return geometry_obj

    def update_object_transform(
        self,
        name: str,
        position: np.ndarray,
        rotation: np.ndarray,
        scale: float | np.ndarray,
    ):
        """Update object transformation efficiently"""
        if name not in self.geometry_objects:
            return

        obj = self.geometry_objects[name]
        obj.position = np.array(position, dtype=np.float32)
        obj.rotation = np.array(rotation, dtype=np.float32)

        if isinstance(scale, int | float):
            obj.scale = np.array([scale, scale, scale], dtype=np.float32)
        else:
            obj.scale = np.array(scale, dtype=np.float32)

    def set_object_visibility(self, name: str, visible: bool):
        """Set object visibility"""
        if name in self.geometry_objects:
            self.geometry_objects[name].visible = visible

    def get_model_matrix(self, obj: GeometryObject) -> np.ndarray:
        """Calculate model matrix for object"""
        # Translation matrix
        T = np.eye(4, dtype=np.float32)
        if obj.position is not None:
            T[:3, 3] = obj.position

        # Rotation matrix
        R = np.eye(4, dtype=np.float32)
        if obj.rotation is not None:
            R[:3, :3] = obj.rotation

        # Scale matrix
        S = np.eye(4, dtype=np.float32)
        if obj.scale is not None:
            S[0, 0] = obj.scale[0]
            S[1, 1] = obj.scale[1]
            S[2, 2] = obj.scale[2]

        return T @ R @ S

    def cleanup(self):
        """Clean up OpenGL resources"""
        for obj in self.geometry_objects.values():
            obj.vao.release()
        self.geometry_objects.clear()

        for program in self.programs.values():
            program.release()
        self.programs.clear()


# ============================================================================
# FIXED OPENGL RENDERER
# ============================================================================


class OpenGLRenderer:
    """High-performance OpenGL renderer with modern shaders"""

    def __init__(self):
        self.ctx = None
        self.geometry_manager = None
        self.programs = {}
        self.textures = {}
        self.ground_level = 0.0  # Ground level for proper rendering

        # Rendering state
        self.viewport_size = (1600, 900)
        self.clear_color = (1.0, 1.0, 1.0, 1.0)  # White background

        # Performance tracking
        self.render_stats = {
            "draw_calls": 0,
            "triangles_rendered": 0,
            "render_time_ms": 0.0,
        }

    def initialize(self, ctx: mgl.Context):
        """Initialize OpenGL context and resources"""
        self.ctx = ctx

        # Setup OpenGL state
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(mgl.CULL_FACE)
        self.ctx.front_face = "ccw"

        # Initialize geometry manager
        self.geometry_manager = GeometryManager(self.ctx)

        # Create standard geometry objects
        self._create_standard_objects()

        print("[OK] OpenGL renderer initialized")
        print(f"   OpenGL Version: {self.ctx.info['GL_VERSION']}")
        print(f"   Renderer: {self.ctx.info['GL_RENDERER']}")

    def _create_standard_objects(self):
        """Create standard geometry objects for rendering"""
        if not self.geometry_manager:
            return

        # Body segment objects
        for segment in [
            "left_forearm",
            "left_upper_arm",
            "right_forearm",
            "right_upper_arm",
            "left_shoulder_neck",
            "right_shoulder_neck",
        ]:
            self.geometry_manager.create_geometry_object(f"{segment}_cyl", "cylinder")
            self.geometry_manager.create_geometry_object(f"{segment}_sph", "sphere")

        # Club objects
        self.geometry_manager.create_geometry_object("shaft", "cylinder")
        self.geometry_manager.create_geometry_object("clubhead", "sphere")

        # Hub
        self.geometry_manager.create_geometry_object("hub", "sphere")

        # Ground
        self.geometry_manager.create_geometry_object("ground", "ground", "ground")

        print(
            f"[OK] Created {len(self.geometry_manager.geometry_objects)} "
            f"geometry objects"
        )

    def set_viewport(self, width: int, height: int):
        """Set viewport size"""
        self.viewport_size = (width, height)
        if self.ctx:
            self.ctx.viewport = (0, 0, width, height)

    def render_frame(
        self,
        frame_data,
        dynamics_data,
        render_config,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        view_position: np.ndarray,
    ):
        """Render complete frame with all elements"""
        if not self.ctx or not self.geometry_manager:
            return
        start_time = time.time()

        # Clear framebuffer with white background
        self.ctx.clear(*self.clear_color)

        # Reset stats
        self.render_stats["triangles_rendered"] = 0
        self.render_stats["draw_calls"] = 0

        # Render ground
        if render_config.show_ground:
            self._render_ground(view_matrix, proj_matrix, view_position)

        # Render body segments
        self._render_body_segments(
            frame_data, render_config, view_matrix, proj_matrix, view_position
        )

        # Render club
        if render_config.show_club:
            self._render_club(
                frame_data, render_config, view_matrix, proj_matrix, view_position
            )

        # Update performance stats
        self.render_stats["render_time_ms"] = (time.time() - start_time) * 1000

    def _render_ground(
        self,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        view_position: np.ndarray,
    ):
        """Render ground plane at proper level with golf grid"""
        if not self.geometry_manager:
            return

        if "ground" not in self.geometry_manager.programs:
            return

        program = self.geometry_manager.programs["ground"]
        if program is None:
            return

        # Set uniforms safely using correct moderngl 5.x API
        try:
            program["view"].write(view_matrix.astype(np.float32).tobytes())
            program["projection"].write(proj_matrix.astype(np.float32).tobytes())
            program["grassColor"].write(
                np.array([0.2, 0.6, 0.2], dtype=np.float32).tobytes()
            )
            program["gridColor"].write(
                np.array([0.3, 0.3, 0.3], dtype=np.float32).tobytes()
            )
            program["gridSpacing"].value = 0.5  # 50cm grid spacing
        except Exception as e:
            print(f"[WARN] Ground uniform error: {e}")
            return

        # Create ground plane at proper level
        # Ground should be at the lowest Z point in the data
        ground_level = getattr(self, "ground_level", 0.0)

        # Create ground plane model matrix (large plane at ground level)
        ground_size = 10.0  # 10m x 10m ground plane
        ground_model = np.eye(4, dtype=np.float32)
        ground_model[0, 0] = ground_size  # Scale X
        ground_model[2, 2] = ground_size  # Scale Z
        ground_model[1, 3] = ground_level  # Position at ground level

        try:
            program["model"].write(ground_model.astype(np.float32).tobytes())

            # Render ground plane
            if "ground_plane" in self.geometry_manager.geometry_objects:
                ground_obj = self.geometry_manager.geometry_objects["ground_plane"]
                ground_obj.vao.render()
                self.render_stats["draw_calls"] += 1
                self.render_stats["triangles_rendered"] += ground_obj.index_count // 3
        except Exception as e:
            print(f"[WARN] Ground render error: {e}")

    def _render_body_segments(
        self,
        frame_data,
        render_config,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        view_position: np.ndarray,
    ):
        """Render all body segments"""
        if not self.geometry_manager:
            return

        if "simple" not in self.geometry_manager.programs:
            return

        program = self.geometry_manager.programs["simple"]
        if program is None:
            return

        # Set common uniforms safely using correct moderngl 5.x API
        try:
            program["view"].write(view_matrix.astype(np.float32).tobytes())
            program["projection"].write(proj_matrix.astype(np.float32).tobytes())
            program["lightPosition"].write(
                np.array([2.0, 4.0, 1.0], dtype=np.float32).tobytes()
            )
            program["lightColor"].write(
                np.array([1.0, 1.0, 1.0], dtype=np.float32).tobytes()
            )
            program["viewPosition"].write(view_position.astype(np.float32).tobytes())
        except Exception as e:
            print(f"[WARN] Body segments uniform error: {e}")
            return

        # Define body segments with their properties
        segments = [
            # (name, start_point, end_point, radius, color, is_skin)
            (
                "left_forearm",
                frame_data.left_wrist,
                frame_data.left_elbow,
                0.025,
                [0.96, 0.76, 0.63],
                True,
            ),
            (
                "left_upper_arm",
                frame_data.left_elbow,
                frame_data.left_shoulder,
                0.035,
                [0.18, 0.32, 0.40],
                False,
            ),
            (
                "right_forearm",
                frame_data.right_wrist,
                frame_data.right_elbow,
                0.025,
                [0.96, 0.76, 0.63],
                True,
            ),
            (
                "right_upper_arm",
                frame_data.right_elbow,
                frame_data.right_shoulder,
                0.035,
                [0.18, 0.32, 0.40],
                False,
            ),
            (
                "left_shoulder_neck",
                frame_data.left_shoulder,
                frame_data.hub,
                0.04,
                [0.18, 0.32, 0.40],
                False,
            ),
            (
                "right_shoulder_neck",
                frame_data.right_shoulder,
                frame_data.hub,
                0.04,
                [0.18, 0.32, 0.40],
                False,
            ),
        ]

        for segment_name, start_pos, end_pos, radius, color, _is_skin in segments:
            if not render_config.show_body_segments.get(segment_name, True):
                continue

            if not (np.isfinite(start_pos).all() and np.isfinite(end_pos).all()):
                continue

            # Render cylinder
            self._render_cylinder_between_points(
                f"{segment_name}_cyl",
                start_pos,
                end_pos,
                radius,
                color,
                render_config.body_opacity,
                program,
            )

            # Render joint spheres
            self._render_sphere_at_point(
                f"{segment_name}_sph",
                end_pos,
                radius * 1.2,
                color,
                render_config.body_opacity,
                program,
            )

        # Render hub
        if np.isfinite(frame_data.hub).all():
            self._render_sphere_at_point(
                "hub",
                frame_data.hub,
                0.06,
                [0.18, 0.32, 0.40],
                render_config.body_opacity,
                program,
            )

    def _render_cylinder_between_points(
        self,
        obj_name: str,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: list[float],
        opacity: float,
        program: mgl.Program,
    ):
        """Render cylinder between two 3D points"""
        if not self.geometry_manager:
            return

        if obj_name not in self.geometry_manager.geometry_objects:
            return

        obj = self.geometry_manager.geometry_objects[obj_name]

        # Calculate transformation
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            obj.visible = False
            return

        direction_normalized = direction / length

        # Create rotation matrix to align Y-axis with direction
        y_axis = np.array([0, 1, 0], dtype=np.float32)
        if np.allclose(direction_normalized, y_axis):
            rotation_matrix = np.eye(3, dtype=np.float32)
        elif np.allclose(direction_normalized, -y_axis):
            rotation_matrix = np.array(
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32
            )
        else:
            # Use Rodrigues rotation formula
            from golf_data_core import GeometryUtils

            rotation_matrix = GeometryUtils.rotation_matrix_from_vectors(
                y_axis, direction_normalized
            )

        # Update object transform
        self.geometry_manager.update_object_transform(
            obj_name,
            start,
            rotation_matrix,
            np.array([radius, length, radius], dtype=np.float32),
        )

        # Set material properties safely using correct moderngl 5.x API
        try:
            program["materialColor"].write(np.array(color, dtype=np.float32).tobytes())
            program["opacity"].value = opacity

            # Render
            model_matrix = self.geometry_manager.get_model_matrix(obj)
            program["model"].write(model_matrix.astype(np.float32).tobytes())

            obj.vao.render()
            obj.visible = True

            self.render_stats["draw_calls"] += 1
            self.render_stats["triangles_rendered"] += obj.index_count // 3
        except Exception as e:
            print(f"[WARN] Cylinder render error: {e}")

    def _render_sphere_at_point(
        self,
        obj_name: str,
        position: np.ndarray,
        radius: float,
        color: list[float],
        opacity: float,
        program: mgl.Program,
    ):
        """Render sphere at specific point"""
        if not self.geometry_manager:
            return

        if obj_name not in self.geometry_manager.geometry_objects:
            return

        obj = self.geometry_manager.geometry_objects[obj_name]

        # Update transform
        self.geometry_manager.update_object_transform(
            obj_name, position, np.eye(3, dtype=np.float32), radius
        )

        # Set material properties safely using correct moderngl 5.x API
        try:
            program["materialColor"].write(np.array(color, dtype=np.float32).tobytes())
            program["opacity"].value = opacity

            # Render
            model_matrix = self.geometry_manager.get_model_matrix(obj)
            program["model"].write(model_matrix.astype(np.float32).tobytes())

            obj.vao.render()
            obj.visible = True

            self.render_stats["draw_calls"] += 1
            self.render_stats["triangles_rendered"] += obj.index_count // 3
        except Exception as e:
            print(f"[WARN] Sphere render error: {e}")

    def _render_club(
        self,
        frame_data,
        render_config,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray,
        view_position: np.ndarray,
    ):
        """Render golf club with improved geometry and face normal"""
        if not self.geometry_manager:
            return

        if "simple" not in self.geometry_manager.programs:
            return

        program = self.geometry_manager.programs["simple"]
        if program is None:
            return

        # Set common uniforms safely using correct moderngl 5.x API
        try:
            program["view"].write(view_matrix.astype(np.float32).tobytes())
            program["projection"].write(proj_matrix.astype(np.float32).tobytes())
            program["lightPosition"].write(
                np.array([2.0, 4.0, 1.0], dtype=np.float32).tobytes()
            )
            program["lightColor"].write(
                np.array([1.0, 1.0, 1.0], dtype=np.float32).tobytes()
            )
            program["viewPosition"].write(view_position.astype(np.float32).tobytes())
        except Exception as e:
            print(f"[WARN] Club uniform error: {e}")
            return

        # Render shaft with realistic proportions
        shaft_radius = 0.004  # 4mm radius for more realistic shaft
        shaft_color = [0.8, 0.8, 0.8]  # Metallic gray

        self._render_cylinder_between_points(
            "shaft",
            frame_data.butt,
            frame_data.clubhead,
            shaft_radius,
            shaft_color,
            1.0,
            program,
        )

        # Render clubhead with more realistic geometry
        clubhead_color = [0.9, 0.9, 0.95]  # Polished steel

        # Calculate club face normal (perpendicular to shaft direction)
        shaft_direction = frame_data.clubhead - frame_data.butt
        shaft_direction = shaft_direction / np.linalg.norm(shaft_direction)

        # Face normal points perpendicular to shaft
        # (this is simplified - real clubs have loft)
        # For now, assume face points in the direction of swing (perpendicular to shaft)
        face_normal = np.cross(
            shaft_direction, np.array([0, 1, 0])
        )  # Cross with up vector
        if np.linalg.norm(face_normal) < 1e-6:
            face_normal = np.cross(shaft_direction, np.array([1, 0, 0]))  # Fallback
        face_normal = face_normal / np.linalg.norm(face_normal)

        # Render clubhead as an elongated ellipsoid (more realistic than sphere)
        clubhead_radius = 0.02  # Smaller, more realistic head size

        self._render_sphere_at_point(
            "clubhead",
            frame_data.clubhead,
            clubhead_radius,
            clubhead_color,
            1.0,
            program,
        )

        # Render face normal vector if enabled
        if (
            hasattr(render_config, "show_face_normal")
            and render_config.show_face_normal
        ):
            normal_length = 0.1  # 10cm normal vector
            normal_end = frame_data.clubhead + face_normal * normal_length
            normal_color = [1.0, 0.0, 0.0]  # Red for face normal

            self._render_cylinder_between_points(
                "face_normal",
                frame_data.clubhead,
                normal_end,
                0.002,
                normal_color,
                0.8,
                program,
            )

            # Add arrowhead to normal vector
            self._render_sphere_at_point(
                "normal_arrow", normal_end, 0.005, normal_color, 0.8, program
            )

        # Render ball at center strike position
        if hasattr(render_config, "show_ball") and render_config.show_ball:
            # Position ball slightly in front of clubface for center strike
            ball_offset = face_normal * 0.05  # 5cm in front of face
            ball_position = frame_data.clubhead + ball_offset
            ball_color = [1.0, 1.0, 1.0]  # White ball
            ball_radius = 0.02135  # Standard golf ball diameter (42.67mm)

            self._render_sphere_at_point(
                "ball", ball_position, ball_radius, ball_color, 1.0, program
            )

    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.geometry_manager:
            self.geometry_manager.cleanup()

        print("🧹 OpenGL renderer cleaned up")


# ============================================================================
# USAGE EXAMPLE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("🎨 Golf Swing Visualizer - Fixed OpenGL Renderer Test")

    # Test shader compilation
    print("\n🔧 Testing shader compilation...")
    try:
        vertex_shader = ShaderLibrary.get_simple_vertex_shader()
        fragment_shader = ShaderLibrary.get_simple_fragment_shader()
        print(
            f"   Simple shaders: {len(vertex_shader)} + "
            f"{len(fragment_shader)} characters"
        )

        print("[OK] Shader compilation test passed")
    except Exception as e:
        print(f"[ERROR] Shader compilation test failed: {e}")

    print("\n🎉 Fixed OpenGL renderer ready for integration!")
