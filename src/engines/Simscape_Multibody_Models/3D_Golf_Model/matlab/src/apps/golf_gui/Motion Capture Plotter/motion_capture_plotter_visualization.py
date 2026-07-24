# mypy: disable-error-code="attr-defined,assignment"
"""Visualization and camera helpers for the legacy motion-capture plotter."""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class MotionCapturePlotterVisualizationMixin:
    _last_pos: tuple[float, float] | None

    def setup_3d_scene(self) -> None:
        """Setup the 3D scene with ground plane and ball."""
        self.ax.clear()

        # Ground plane - positioned at calculated ground level
        ground_level = self.calculate_ground_level()
        x_ground = np.linspace(-3, 3, 10)
        y_ground = np.linspace(-3, 3, 10)
        X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
        Z_ground = np.full_like(X_ground, ground_level)  # Set to actual ground level
        self.ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.3, color="green")

        # Golf ball - positioned at origin for golf swing analysis
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        ball_radius = 0.021
        x_ball = ball_radius * np.outer(np.cos(u), np.sin(v))
        y_ball = ball_radius * np.outer(np.sin(u), np.sin(v))
        z_ball = ball_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_surface(x_ball, y_ball, z_ball, color="white", alpha=0.8)

        # Set axis labels and limits
        self.ax.set_xlabel("X (Target Line)")
        self.ax.set_ylabel("Y (Ball Direction)")
        self.ax.set_zlabel("Z (Vertical)")
        self.ax.set_xlim([-2.0, 2.0])
        self.ax.set_ylim([-1.0, 3.0])

        # Set initial Z limits - will be adjusted based on data
        self.ax.set_zlim([-0.5, 2.5])

        # Set initial view
        self.ax.view_init(elev=15, azim=-45)

    def calculate_ground_level(self) -> float:
        """Set ground level to fixed -2.5 meters."""
        return -2.5

    def adjust_plot_limits_to_ground(self) -> None:
        """Adjust plot limits so ground level is at the bottom."""
        ground_level = self.calculate_ground_level()

        # Set Z limits with ground at bottom and reasonable height above
        z_range = 3.0  # 3 meters total height
        self.ax.set_zlim([ground_level, ground_level + z_range])

        logger.info(f"Ground level set to: {ground_level:.3f}m")

    def update_visualization(self) -> None:
        """Update the 3D visualization with proper coordinate system."""
        # Store current view angles before clearing
        current_elev = self.ax.elev
        current_azim = self.ax.azim

        # Clear and setup scene
        self.setup_3d_scene()

        # Adjust ground level based on data
        self.adjust_plot_limits_to_ground()

        # Restore the view angles to maintain user's camera position
        self.ax.view_init(elev=current_elev, azim=current_azim)

        # Visualize motion capture data if enabled
        if self.show_motion_capture and self.swing_data:
            # Find the first available swing data
            available_swings = list(self.swing_data.keys())
            if available_swings and self.current_frame < len(
                self.swing_data[available_swings[0]]
            ):  # noqa: E501
                motion_data = self.swing_data[available_swings[0]]
                frame_data = motion_data.iloc[self.current_frame]
                self.visualize_motion_capture_data(frame_data, motion_data)

        # Visualize Simscape data if enabled
        if self.show_simscape and self.simscape_data:
            # Find the first available Simscape data
            available_simscape = list(self.simscape_data.keys())
            if available_simscape and self.current_frame < len(
                self.simscape_data[available_simscape[0]]
            ):
                simscape_data = self.simscape_data[available_simscape[0]]
                frame_data = simscape_data.iloc[self.current_frame]
                self.visualize_simscape_data(frame_data, simscape_data)

        # Update info text with combined data
        self.update_info_text(
            None
        )  # Pass None since we're handling multiple data sources  # noqa: E501

        # Redraw canvas
        self.canvas.draw()

    def _draw_motion_capture_trajectory_paths(self, data) -> None:
        """Draw mid-hands and club head trajectory paths for motion capture data.

        Parameters:
            data: full DataFrame of all frames
        """
        if data is None:
            raise ValueError("data must be provided")
        if self.trajectory_check.isChecked() and len(data) > 1:
            # Mid-hands path (blue dashed) - flip X for right-handed swing
            # ⚡ Bolt: Vectorized column stack is >1000x faster than iterrows()
            trajectory = np.column_stack(
                (
                    -data["mid_X"].values * self.motion_scale,
                    data["mid_Y"].values * self.motion_scale,
                    data["mid_Z"].values * self.motion_scale,
                )
            )
            self.ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                "b--",
                alpha=0.6,
                linewidth=2,
                label="Mid-Hands Path",
            )

        if self.club_path_check.isChecked() and len(data) > 1:
            # Club head path (red dashed) - flip X for right-handed swing
            # ⚡ Bolt: Vectorized column stack is >1000x faster than iterrows()
            club_path = np.column_stack(
                (
                    -data["club_X"].values * self.motion_scale,
                    data["club_Y"].values * self.motion_scale,
                    data["club_Z"].values * self.motion_scale,
                )
            )
            self.ax.plot(
                club_path[:, 0],
                club_path[:, 1],
                club_path[:, 2],
                "r--",
                alpha=0.6,
                linewidth=2,
                label="Club Head Path",
            )

    def visualize_motion_capture_data(self, frame_data, data) -> None:
        """Visualize motion capture data (Excel format)."""
        # Use actual mid-hands and club head positions from the data
        # For right-handed golfers: X should be flipped to show proper swing direction
        if frame_data is None:
            raise ValueError("frame_data must be provided")
        mid_hands = np.array(
            [
                -frame_data["mid_X"]
                * self.motion_scale,  # Flip X for right-handed swing  # noqa: E501
                frame_data["mid_Y"] * self.motion_scale,
                frame_data["mid_Z"] * self.motion_scale,
            ]
        )

        club_head = np.array(
            [
                -frame_data["club_X"]
                * self.motion_scale,  # Flip X for right-handed swing  # noqa: E501
                frame_data["club_Y"] * self.motion_scale,
                frame_data["club_Z"] * self.motion_scale,
            ]
        )

        # Draw the club shaft, head, face normal, and golf ball
        self._draw_club_with_face_normal(club_head, mid_hands)

        # Draw trajectory paths
        self._draw_motion_capture_trajectory_paths(data)

    def _extract_joint_positions(self, frame_data) -> dict[str, np.ndarray]:
        """Extract scaled joint positions from a Simscape frame.

        Returns a dict mapping joint names to numpy position arrays.
        """
        if frame_data is None:
            raise ValueError("frame_data must be provided")
        joints = {}
        joint_names = [
            "club_head",
            "left_hand",
            "right_hand",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "hub",
            "spine",
            "hip",
        ]
        for joint_name in joint_names:
            if f"{joint_name}_X" in frame_data:
                joints[joint_name] = np.array(
                    [
                        -frame_data[f"{joint_name}_X"]
                        * self.motion_scale,  # Flip X for right-handed swing
                        frame_data[f"{joint_name}_Y"] * self.motion_scale,
                        frame_data[f"{joint_name}_Z"] * self.motion_scale,
                    ]
                )
        return joints

    def _draw_club_with_face_normal(self, club_head_pos, grip_pos) -> None:
        """Draw the club shaft, head sphere, face normal vector, and golf ball.

        Parameters:
            club_head_pos: numpy array of club head position [x, y, z]
            grip_pos: numpy array of grip (left hand) position [x, y, z]
        """
        # Draw club shaft from grip to club head
        if club_head_pos is None:
            raise ValueError("club_head_pos must be provided")
        club_points = np.array([grip_pos, club_head_pos])
        self.ax.plot(
            club_points[:, 0],
            club_points[:, 1],
            club_points[:, 2],
            color="gray",
            linewidth=6,
            alpha=0.9,
            label="Club Shaft",
        )

        # Draw club head as sphere with better appearance
        u = np.linspace(0, 2 * np.pi, 8)
        v = np.linspace(0, np.pi, 8)
        head_size = 0.03  # Realistic club head size
        x_head = head_size * np.outer(np.cos(u), np.sin(v)) + club_head_pos[0]
        y_head = head_size * np.outer(np.sin(u), np.sin(v)) + club_head_pos[1]
        z_head = head_size * np.outer(np.ones(np.size(u)), np.cos(v)) + club_head_pos[2]
        self.ax.plot_surface(
            x_head, y_head, z_head, color="darkgray", alpha=0.9, label="Club Head"
        )  # noqa: E501

        # Calculate and draw club face normal vector
        shaft_direction = club_head_pos - grip_pos
        # ⚡ Bolt: math.hypot is faster than np.linalg.norm for small 1D arrays
        shaft_length = math.hypot(
            shaft_direction[0], shaft_direction[1], shaft_direction[2]
        )

        if shaft_length > 0:
            shaft_direction = shaft_direction / shaft_length

            # Calculate face normal (perpendicular to shaft)
            up_vector = np.array([0, 0, 1])  # Vertical up
            face_normal = np.cross(shaft_direction, up_vector)
            # ⚡ Bolt: math.hypot is faster than np.linalg.norm for small 1D arrays
            face_normal_length = math.hypot(
                face_normal[0], face_normal[1], face_normal[2]
            )

            if face_normal_length > 0:
                face_normal = face_normal / face_normal_length
                self._draw_face_normal_and_ball(club_head_pos, face_normal)

    def _draw_face_normal_and_ball(self, club_head_pos, face_normal) -> None:
        """Draw the face normal vector arrow and a golf ball in front of the club.

        Parameters:
            club_head_pos: numpy array of club head position [x, y, z]
            face_normal: unit numpy array of face normal direction
        """
        # Draw face normal vector (red arrow) - longer and more visible
        if club_head_pos is None:
            raise ValueError("club_head_pos must be provided")
        normal_length = 0.25  # 25cm normal vector (longer)
        normal_end = club_head_pos + face_normal * normal_length

        # Draw the normal vector as a thick line
        normal_points = np.array([club_head_pos, normal_end])
        self.ax.plot(
            normal_points[:, 0],
            normal_points[:, 1],
            normal_points[:, 2],
            "red",
            linewidth=6,
            alpha=1.0,
            label="Face Normal",
        )

        # Add a larger arrowhead at the end
        self.ax.scatter(
            normal_end[0],
            normal_end[1],
            normal_end[2],
            c="red",
            s=200,
            marker=">",
            alpha=1.0,
        )

        # Add a small sphere at the start of the normal for better visibility
        self.ax.scatter(
            club_head_pos[0],
            club_head_pos[1],
            club_head_pos[2],
            c="red",
            s=50,
            marker="o",
            alpha=0.8,
        )

        # Draw golf ball positioned for center strike
        ball_offset_distance = 0.08  # 8cm in front of club face
        ball_position = club_head_pos + face_normal * ball_offset_distance

        # Draw golf ball as a white sphere
        ball_radius = 0.021  # Standard golf ball radius
        u_ball = np.linspace(0, 2 * np.pi, 12)
        v_ball = np.linspace(0, np.pi, 12)
        x_ball = (
            ball_radius * np.outer(np.cos(u_ball), np.sin(v_ball)) + ball_position[0]
        )  # noqa: E501
        y_ball = (
            ball_radius * np.outer(np.sin(u_ball), np.sin(v_ball)) + ball_position[1]
        )  # noqa: E501
        z_ball = (
            ball_radius * np.outer(np.ones(np.size(u_ball)), np.cos(v_ball))
            + ball_position[2]
        )  # noqa: E501
        self.ax.plot_surface(
            x_ball,
            y_ball,
            z_ball,
            color="white",
            alpha=0.95,
            edgecolor="lightgray",
            linewidth=0.5,
            label="Golf Ball",
        )

    def _draw_body_segments_and_markers(self, joints, segment_definitions) -> None:
        """Draw body segment lines and joint marker dots.

        Parameters:
            joints: dict mapping joint names to numpy position arrays
            segment_definitions: list of (start_joint, end_joint, color) tuples
        """
        # Draw body segments
        if joints is None:
            raise ValueError("joints must be provided")
        for start_joint, end_joint, color in segment_definitions:
            if start_joint in joints and end_joint in joints:
                segment_points = np.array([joints[start_joint], joints[end_joint]])
                self.ax.plot(
                    segment_points[:, 0],
                    segment_points[:, 1],
                    segment_points[:, 2],
                    color=color,
                    linewidth=3,
                    alpha=0.7,
                )

        # Draw joint markers
        for _, position in joints.items():
            self.ax.scatter(
                position[0], position[1], position[2], color="black", s=50, alpha=0.8
            )  # noqa: E501

    def _draw_simscape_trajectory_paths(self, joints, data) -> None:
        """Draw club head and hands trajectory paths for Simscape data.

        Parameters:
            joints: dict mapping joint names to numpy position arrays
            data: full DataFrame of all frames
        """
        # Club head trajectory
        if joints is None:
            raise ValueError("joints must be provided")
        if (
            self.trajectory_check.isChecked()
            and len(data) > 1
            and "club_head" in joints
        ):  # noqa: E501
            # ⚡ Bolt: Vectorized column stack is >1000x faster than iterrows()
            club_trajectory = np.column_stack(
                (
                    -data["club_head_X"].values * self.motion_scale,
                    data["club_head_Y"].values * self.motion_scale,
                    data["club_head_Z"].values * self.motion_scale,
                )
            )
            if len(club_trajectory) > 1:
                self.ax.plot(
                    club_trajectory[:, 0],
                    club_trajectory[:, 1],
                    club_trajectory[:, 2],
                    "r--",
                    alpha=0.6,
                    linewidth=2,
                    label="Club Head Path",
                )

            # Hands trajectory
        if self.club_path_check.isChecked() and len(data) > 1 and "left_hand" in joints:
            # ⚡ Bolt: Vectorized column stack is >1000x faster than iterrows()
            hands_trajectory = np.column_stack(
                (
                    -data["left_hand_X"].values * self.motion_scale,
                    data["left_hand_Y"].values * self.motion_scale,
                    data["left_hand_Z"].values * self.motion_scale,
                )
            )
            if len(hands_trajectory) > 1:
                self.ax.plot(
                    hands_trajectory[:, 0],
                    hands_trajectory[:, 1],
                    hands_trajectory[:, 2],
                    "b--",
                    alpha=0.6,
                    linewidth=2,
                    label="Hands Path",
                )

    def _draw_segment_traces(self, frame_data, data) -> None:
        """Draw optional per-segment trace paths for Simscape data.

        Parameters:
            frame_data: current frame's data row
            data: full DataFrame of all frames
        """
        if frame_data is None:
            raise ValueError("frame_data must be provided")
        trace_colors = {
            "club_head": "red",
            "left_hand": "blue",
            "right_hand": "cyan",
            "left_elbow": "green",
            "right_elbow": "lime",
            "left_shoulder": "orange",
            "right_shoulder": "yellow",
            "hub": "purple",
            "spine": "magenta",
            "hip": "brown",
        }

        for segment_key, checkbox in self.segment_traces.items():
            if (
                checkbox.isChecked()
                and f"{segment_key}_X" in frame_data
                and len(data) > 1
            ):  # noqa: E501
                # Create trajectory for this segment
                # ⚡ Bolt: Vectorized column stack is >1000x faster than iterrows()
                segment_trajectory = (
                    np.column_stack(
                        (
                            -data[f"{segment_key}_X"].values * self.motion_scale,
                            data[f"{segment_key}_Y"].values * self.motion_scale,
                            data[f"{segment_key}_Z"].values * self.motion_scale,
                        )
                    )
                    if f"{segment_key}_X" in data.columns
                    else np.empty((0, 3))
                )
                if len(segment_trajectory) > 1:
                    color = trace_colors.get(segment_key, "gray")
                    self.ax.plot(
                        segment_trajectory[:, 0],
                        segment_trajectory[:, 1],
                        segment_trajectory[:, 2],
                        color=color,
                        linestyle="--",
                        alpha=0.6,
                        linewidth=2,
                        label=f"{segment_key.replace('_', ' ').title()} Path",
                    )

    def visualize_simscape_data(self, frame_data, data) -> None:
        """Visualize Simscape multibody data (CSV format)."""
        # Define colors for different body segments
        if frame_data is None:
            raise ValueError("frame_data must be provided")
        colors = {
            "club": "red",
            "hands": "blue",
            "arms": "green",
            "shoulders": "orange",
            "torso": "purple",
            "hips": "brown",
        }

        joints = self._extract_joint_positions(frame_data)

        # Define segments connecting joints
        segment_definitions = [
            ("left_hand", "right_hand", colors["hands"]),  # Midpoint
            ("left_hand", "left_elbow", colors["arms"]),
            ("right_hand", "right_elbow", colors["arms"]),
            ("left_elbow", "left_shoulder", colors["arms"]),
            ("right_elbow", "right_shoulder", colors["arms"]),
            ("left_shoulder", "right_shoulder", colors["shoulders"]),
            ("left_shoulder", "hub", colors["torso"]),
            ("right_shoulder", "hub", colors["torso"]),
            ("hub", "spine", colors["torso"]),
            ("spine", "hip", colors["torso"]),
        ]

        # Draw club if available
        if "club_head" in joints and "left_hand" in joints:
            self._draw_club_with_face_normal(joints["club_head"], joints["left_hand"])

        self._draw_body_segments_and_markers(joints, segment_definitions)
        self._draw_simscape_trajectory_paths(joints, data)
        self._draw_segment_traces(frame_data, data)

    def update_info_text(self, frame_data) -> None:
        """Update the information text display."""
        info = f"Frame: {self.current_frame}\n"
        info += f"Data Source: {self.current_data_source}\n"
        info += f"Motion Scale: {self.motion_scale}x\n\n"

        # Show motion capture data if available
        if self.show_motion_capture and self.swing_data:
            available_swings = list(self.swing_data.keys())
            if available_swings and self.current_frame < len(
                self.swing_data[available_swings[0]]
            ):  # noqa: E501
                motion_data = self.swing_data[available_swings[0]]
                motion_frame = motion_data.iloc[self.current_frame]
                info += "Motion Capture Data:\n"
                info += f"  Time: {motion_frame['time']:.3f}s\n"
                info += (
                    f"  Mid-Hands: ({motion_frame['mid_X']:.3f}, "
                    f"{motion_frame['mid_Y']:.3f}, "
                    f"{motion_frame['mid_Z']:.3f})\n"
                )
                info += (
                    f"  Club Head: ({motion_frame['club_X']:.3f}, "
                    f"{motion_frame['club_Y']:.3f}, "
                    f"{motion_frame['club_Z']:.3f})\n\n"
                )

        # Show Simscape data if available
        if self.show_simscape and self.simscape_data:
            available_simscape = list(self.simscape_data.keys())
            if available_simscape and self.current_frame < len(
                self.simscape_data[available_simscape[0]]
            ):
                simscape_data = self.simscape_data[available_simscape[0]]
                simscape_frame = simscape_data.iloc[self.current_frame]
                info += "Simscape Data:\n"
                info += f"  Time: {simscape_frame['time']:.3f}s\n"
                info += "  Available Joints:\n"
                joint_count = 0
                for joint_name in [
                    "club_head",
                    "left_hand",
                    "right_hand",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "hub",
                    "spine",
                    "hip",
                ]:
                    if f"{joint_name}_X" in simscape_frame:
                        info += f"    {joint_name}: ✓\n"
                        joint_count += 1
                    else:
                        info += f"    {joint_name}: ✗\n"
                info += f"\nTotal Joints: {joint_count}"

        self.info_text.setText(info)

    def set_camera_view(self, view) -> None:
        """Set predefined camera views."""
        if view is None:
            raise ValueError("view must be provided")
        if view == "face_on":
            # Face-on view: looking at golfer from front (toward +X target line)
            self.ax.view_init(elev=15, azim=90)
        elif view == "down_line":
            # Down-the-line view: looking from side (90° from face-on)
            self.ax.view_init(elev=15, azim=180)
        elif view == "top_down":
            # Top-down view: looking down from above
            self.ax.view_init(elev=90, azim=0)
        elif view == "isometric":
            # Isometric view: 3D perspective
            self.ax.view_init(elev=15, azim=-45)

        # Force redraw
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def reset_view(self) -> None:
        """Reset the 3D view to the default isometric view and limits."""
        # Reset view angles
        self.ax.view_init(elev=15, azim=-45)

        # Reset plot limits to default
        self.ax.set_xlim([-2.0, 2.0])
        self.ax.set_ylim([-1.0, 3.0])
        self.ax.set_zlim([-0.5, 2.5])

        # Force redraw
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def on_scroll(self, event) -> None:
        """Handle mouse scroll for zooming."""
        if event is None:
            raise ValueError("event must be provided")
        if event.inaxes != self.ax:
            return

        logger.info(f"Scroll event: button={event.button}, step={event.step}")

        # Get current view limits
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
        z_lim = self.ax.get_zlim()

        # Determine zoom factor based on scroll direction
        zoom_factor = 0.9 if event.button == "up" or event.step > 0 else 1.1  # Zoom out

        # Calculate centers
        x_center = (x_lim[0] + x_lim[1]) / 2
        y_center = (y_lim[0] + y_lim[1]) / 2
        z_center = (z_lim[0] + z_lim[1]) / 2

        # Calculate new ranges
        x_range = (x_lim[1] - x_lim[0]) * zoom_factor
        y_range = (y_lim[1] - y_lim[0]) * zoom_factor
        z_range = (z_lim[1] - z_lim[0]) * zoom_factor

        # Set new limits
        self.ax.set_xlim([x_center - x_range / 2, x_center + x_range / 2])
        self.ax.set_ylim([y_center - y_range / 2, y_center + y_range / 2])
        self.ax.set_zlim([z_center - z_range / 2, z_center + z_range / 2])

        self.canvas.draw()
        logger.info(f"Zooming: factor={zoom_factor}")

    def on_mouse_press(self, event) -> None:
        """Handle mouse button press for rotation/panning."""
        if event is None:
            raise ValueError("event must be provided")
        if event.inaxes != self.ax:
            return
        # Store initial position for rotation/panning (use screen coordinates)
        self._last_pos = (event.x, event.y)
        logger.info(f"Mouse press: button={event.button}, pos=({event.x}, {event.y})")

    def on_mouse_release(self, event) -> None:
        """Handle mouse button release."""
        self._last_pos = None

    def on_mouse_move(self, event) -> None:
        """Handle mouse movement for rotation/panning."""
        if event is None:
            raise ValueError("event must be provided")
        if event.inaxes != self.ax or self._last_pos is None:
            return

        if hasattr(event, "button") and event.button == 1:  # Left click - rotate
            # Get current view angles
            elev = self.ax.elev
            azim = self.ax.azim

            # Calculate change in position (use screen coordinates for better control)
            dx = event.x - self._last_pos[0]
            dy = event.y - self._last_pos[1]

            # Update view angles (scale the movement)
            self.ax.view_init(elev=elev + dy * 0.5, azim=azim + dx * 0.5)
            self.canvas.draw()
            logger.info(
                f"Rotating: dx={dx}, dy={dy}, "
                f"new_elev={elev + dy * 0.5}, new_azim={azim + dx * 0.5}"
            )

        elif hasattr(event, "button") and event.button == 3:  # Right click - pan
            # Get current limits
            x_lim = self.ax.get_xlim()
            y_lim = self.ax.get_ylim()
            z_lim = self.ax.get_zlim()

            # Calculate change in position (use screen coordinates)
            dx = event.x - self._last_pos[0]
            dy = event.y - self._last_pos[1]

            # Update limits (scale the movement)
            x_range = x_lim[1] - x_lim[0]
            y_range = y_lim[1] - y_lim[0]
            z_lim[1] - z_lim[0]

            pan_scale = 0.01  # Adjust this for panning sensitivity

            self.ax.set_xlim(
                [
                    x_lim[0] - dx * x_range * pan_scale,
                    x_lim[1] - dx * x_range * pan_scale,
                ]
            )
            self.ax.set_ylim(
                [
                    y_lim[0] + dy * y_range * pan_scale,
                    y_lim[1] + dy * y_range * pan_scale,
                ]
            )
            self.canvas.draw()
            logger.info(f"Panning: dx={dx}, dy={dy}")

        self._last_pos = (event.x, event.y)
