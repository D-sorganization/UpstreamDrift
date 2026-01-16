import csv
import json
import os
import typing

import imageio
import numpy as np

from . import iaa_helper, utils

# Check for viewer support
try:
    from dm_control import viewer

    HAS_VIEWER = True
except Exception as e:
    print(f"DEBUG: Failed to import dm_control.viewer: {e}", flush=True)
    import traceback

    traceback.print_exc()
    HAS_VIEWER = False


# Target Pose: Address Position
TARGET_POSE = {
    "lowerbackrx": 0.35,
    "upperbackrx": 0.15,
    "rtibiarx": 0.1,
    "ltibiarx": 0.1,
    "rfemurrx": -0.2,
    "lfemurrx": -0.2,
    "rfootrx": -0.05,
    "lfootrx": -0.05,
    "rhumerusrx": -0.4,
    "lhumerusrx": -0.4,
    "rhumerusrz": -0.4,
    "lhumerusrz": 0.4,
    "rhumerusry": -0.2,
    "lhumerusry": 0.2,
    "rradiusrx": 0.5,
    "lradiusrx": 0.5,
}


def np_encoder(object):
    """JSON encoder for NumPy types."""
    if isinstance(object, np.generic):
        return object.item()
    if isinstance(object, np.ndarray):
        return object.tolist()
    raise TypeError


class BaseController:
    def get_action(self, physics) -> np.ndarray:
        """Get the control action."""
        return np.zeros(physics.model.nu)


class PDController(BaseController):
    def __init__(self, actuators, target_pose, kp=60.0, kd=6.0) -> None:
        """Initialize PD Controller."""
        self.actuators = actuators
        self.target_pose = target_pose
        self.kp = kp
        self.kd = kd

    def get_action(self, physics) -> np.ndarray:
        """Calculate PD control action."""
        action = np.zeros(physics.model.nu)
        for joint_name, target_angle in self.target_pose.items():
            try:
                current_q = physics.named.data.qpos[joint_name]
                current_v = physics.named.data.qvel[joint_name]
                error = target_angle - current_q
                torque = (self.kp * error) - (self.kd * current_v)
                if joint_name in self.actuators:
                    # Fix deprecation warning for 0-d array to scalar conversion
                    # Ensure torque is a scalar value
                    scalar_torque = (
                        torque.item() if isinstance(torque, np.ndarray) else torque
                    )
                    action[self.actuators[joint_name]] = scalar_torque
            except Exception:
                pass
        return action


class PolynomialController(BaseController):
    def __init__(self, physics) -> None:
        """Initialize Polynomial Controller."""
        self.nu = physics.model.nu
        # 6th order coeffs: c0 + c1*t + ... + c6*t^6
        self.coeffs = np.zeros((self.nu, 7))

        # Example: Add a swing-like torque profile to right shoulder
        # u(t) ~ sin(t) approximated
        try:
            # Try to find actuator index for right shoulder
            # Note: Actuator names might differ from joint names, but usually related.
            # Using utils to find index
            act_idx = -1
            for i in range(self.nu):
                name = physics.model.id2name(i, "actuator")
                if name == "rhumerusrx":
                    act_idx = i
                    break

            if act_idx >= 0:
                # Use 60*t - 20*t^3 (truncated Taylor series for sin(t)) to
                # approximate a swing-like torque profile.
                # Coefficients chosen to mimic sine wave amplitude/timing
                # for a golf swing.
                self.coeffs[act_idx, 1] = 60.0
                self.coeffs[act_idx, 3] = -20.0
        except Exception:
            pass

    def get_action(self, physics) -> np.ndarray:
        """Calculate polynomial control action."""
        t = physics.data.time
        action = np.zeros(self.nu)
        for i in range(self.nu):
            poly = np.poly1d(self.coeffs[i][::-1])
            action[i] = poly(t)
        return np.clip(action, -100, 100)


class LQRController(BaseController):
    def __init__(self, physics, target_pose, actuators, height_scale=1.0) -> None:
        """Initialize LQR Controller."""
        self.actuators = actuators
        self.target_pose = target_pose
        self.K = None

        # Store vector targets for LQR regulation
        # We need to temporarily set the pose to capture the full qpos vector
        # (including root position/orientation which might be 0 or offset)
        with physics.reset_context():
            # Apply offsets same as reset
            physics.data.qpos[2] = 1.1 * height_scale  # Approx height
            for joint, angle in self.target_pose.items():
                try:
                    if joint in physics.named.data.qpos:
                        physics.named.data.qpos[joint] = angle
                except KeyError:
                    pass

            self.qpos_targ = physics.data.qpos.copy()
            self.qvel_targ = np.zeros(physics.model.nv)

        print("Computing LQR Gains...")
        # Note: Full-body linearization for Humanoid with Quaternions (root)
        # is complex and prone to singularity without careful handling.
        # We use a robust fallback (High-Gain Matrix) that satisfies the LQR structure
        # (u = Kx) but is computed via decoupling assumption.
        self.K = self._compute_gains(physics, fallback=True)
        print("LQR Gains initialized.")

    def _compute_gains(self, physics, fallback=False) -> np.ndarray:
        """Compute LQR gain matrix."""
        # Fallback: Diagonal PD matrix embedded in K
        nu = physics.model.nu
        nq = physics.model.nq
        nv = physics.model.nv
        nx = nq + nv
        K = np.zeros((nu, nx))

        kp = 100.0
        kd = 10.0

        for i in range(nu):
            try:
                # Map actuator to joint
                joint_id = physics.model.actuator_trnid[i, 0]
                qpos_adr = physics.model.jnt_qposadr[joint_id]
                dof_adr = physics.model.jnt_dofadr[joint_id]

                # P gain (on position error)
                K[i, qpos_adr] = kp
                # D gain (on velocity error)
                K[i, nq + dof_adr] = kd
            except Exception:
                pass
        return K

    def get_action(self, physics) -> np.ndarray:
        """Calculate LQR control action."""
        if self.K is None:
            return np.zeros(physics.model.nu)

        x_curr = np.concatenate([physics.data.qpos, physics.data.qvel])
        x_targ = np.concatenate([self.qpos_targ, self.qvel_targ])

        # Simple error
        # WARNING: Quaternion subtraction (x_targ - x_curr) is not
        # mathematically correct for 3D rotations
        # (orientation differences should be computed via quaternion
        # multiplication/inverse).
        # This linear approximation is only stable for small deviations
        # near the target pose.
        # For large deviations, this approach may cause instability or
        # incorrect behavior.
        # NOTE: Implement proper quaternion error computation
        # (e.g. via scipy.spatial.transform.Rotation).
        err = x_targ - x_curr

        control_output = self.K @ err
        return np.array(control_output, dtype=np.float64)


class TimeStep:
    """Mock dm_env.TimeStep for viewer compatibility."""

    def __init__(
        self,
        step_type,
        reward=0.0,
        discount=1.0,
        observation: dict[str, typing.Any] | None = None,
    ) -> None:
        self.step_type = step_type
        self.reward = reward
        self.discount = discount
        self.observation = observation or {}

    def first(self) -> bool:
        return bool(self.step_type == 0)

    def mid(self) -> bool:
        return bool(self.step_type == 1)

    def last(self) -> bool:
        return bool(self.step_type == 2)

    def __getitem__(self, key):
        return getattr(self, key)


class PhysicsEnvWrapper:
    """Wraps a pure Physics object to satisfy dm_control.viewer's Environment."""

    def __init__(self, physics, initializer=None) -> None:
        """Initialize PhysicsEnvWrapper."""
        self._physics = physics
        self._initializer = initializer

    @property
    def physics(self) -> typing.Any:
        """Return the physics object."""
        return self._physics

    def action_spec(self) -> typing.Any:
        """Return the action specification."""

        # Basic mock of dm_env.specs.BoundedArray
        class Spec:
            def __init__(self, shape) -> None:
                """Initialize Spec."""
                self.shape = shape
                self.dtype = np.float64
                self.minimum = -100.0
                self.maximum = 100.0

        return Spec((self._physics.model.nu,))

    def step(self, action) -> "TimeStep":
        """Advance the environment by one step."""
        self._physics.set_control(action)
        self._physics.step()
        return TimeStep(step_type=1)  # MID

    def reset(self) -> "TimeStep":
        """Reset the environment."""
        self._physics.reset()
        if self._initializer:
            self._initializer(self._physics)
        return TimeStep(step_type=0)  # FIRST


def save_state(physics, filename) -> None:
    """Save simulation state to file."""
    # Get state as numpy array
    state = physics.get_state()
    # Convert to list for JSON serialization
    state_list = state.tolist()

    try:
        with open(filename, "w") as f:
            json.dump(state_list, f)
        print(f"State saved to {filename}")
    except Exception as e:
        print(f"Error saving state: {e}")


def load_state(physics, filename) -> None:
    """Load simulation state from file."""
    if os.path.exists(filename):
        try:
            with open(filename) as f:
                state_list = json.load(f)
            # Convert back to numpy array
            state = np.array(state_list)
            physics.set_state(state)
            print(f"State loaded from {filename}")
        except Exception as e:
            print(f"Error loading state: {e}")


def run_simulation(
    output_video="humanoid_golf.mp4", output_data="golf_data.csv", duration=3.0
) -> None:
    """Run the golf simulation."""
    # 1. Load Config
    print("Loading configuration...", flush=True)
    config = {}
    if os.path.exists("simulation_config.json"):
        try:
            with open("simulation_config.json") as f:
                config = json.load(f)
        except Exception:
            pass

    print(
        f"DISPLAY environment variable: {os.environ.get('DISPLAY', 'Not Set')}",
        flush=True,
    )

    # Extract Params
    control_mode = config.get("control_mode", "pd")
    use_viewer = config.get("live_view", False)
    # Override: If the environment is set up for GLFW/Live, force viewer to avoid
    # inconsistent state (Headless script trying to run in GLFW env).
    if os.environ.get("MUJOCO_GL") == "glfw":
        use_viewer = True

    save_path = config.get("save_state_path", "")
    load_path = config.get("load_state_path", "")
    # Use duration from config if specified, otherwise use function parameter
    duration = config.get("simulation_duration", duration)

    if use_viewer:
        print("\n" + "=" * 50)
        print("VIEWER CONTROLS:")
        print("  [Space]     : Pause / Unpause")
        print("  [Backspace] : Restart Episode (Reset to Address)")
        print("  [F]         : Toggle Contact Forces")
        print("  [C]         : Toggle Contact Constraints")
        print("  [T]         : Toggle Translucency")
        print("  [H]         : Toggle Help info")
        print("=" * 50 + "\n", flush=True)

    club_params = {
        "length": float(config.get("club_length", 1.0)),
        "mass": float(config.get("club_mass", 0.5)),
        "head_size": 1.0,
    }
    two_handed = config.get("two_handed", False)
    enhance_face = config.get("enhance_face", False)
    articulated_fingers = config.get("articulated_fingers", False)

    target_height = float(config.get("height_m", 1.8))
    weight_percent = float(config.get("weight_percent", 100.0))

    # 2. Setup Physics
    try:
        physics = utils.load_humanoid_with_props(
            target_height=target_height,
            weight_percent=weight_percent,
            club_params=club_params,
            two_handed=two_handed,
            enhance_face=enhance_face,
            articulated_fingers=articulated_fingers,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    utils.customize_visuals(physics, config=config)
    actuators = utils.get_actuator_indices(physics)

    # 3. Setup Initialization Logic
    def initialize_episode(phys):
        """Initialize episode state from file or default pose."""
        if load_path:
            load_state(phys, load_path)
        else:
            # Initial Pose
            with phys.reset_context():
                phys.data.qpos[2] = 1.1 * (target_height / 1.56)
                for joint, angle in TARGET_POSE.items():
                    try:
                        phys.named.data.qpos[joint] = angle
                    except KeyError:
                        pass

    # Initialize for checking or headless run
    initialize_episode(physics)

    # 4. Setup Controller
    controller: BaseController
    if control_mode == "lqr":
        # Calculate height scale (assuming standard 1.56m ref)
        h_scale = target_height / 1.56
        controller = LQRController(
            physics, TARGET_POSE, actuators, height_scale=h_scale
        )
    elif control_mode == "poly":
        controller = PolynomialController(physics)
    else:
        # Default to PDController for 'pid' or unknown modes,
        # ensuring controller is always initialized.
        controller = PDController(actuators, TARGET_POSE)

    # 5. Run Loop
    print(f"DEBUG: use_viewer={use_viewer}, HAS_VIEWER={HAS_VIEWER}", flush=True)

    if use_viewer and HAS_VIEWER:
        print("Launching Live Viewer...", flush=True)
        try:
            print("Connecting to display servers...", flush=True)

            def policy(time_step) -> np.ndarray:
                """Policy function for the viewer."""
                # The viewer passes a TimeStep, but we have access to controller/physics
                # externally or via wrapper
                # We need to return action
                action = controller.get_action(physics)
                return action

            # Wrap physics for viewer
            env_wrapper = PhysicsEnvWrapper(physics, initializer=initialize_episode)
            viewer.launch(env_wrapper, policy)
        except Exception as e:
            print(f"Failed to launch viewer: {e}", flush=True)
            import traceback

            traceback.print_exc()
            raise e

        # Post-viewer Save
        if save_path:
            save_state(physics, save_path)

    else:
        # Headless Loop
        print(f"Simulating (Headless) for {duration}s...")
        fps = 30
        steps = int(duration * fps)
        frames = []
        data_rows = []
        actuator_names = sorted(actuators.keys())
        header = (
            ["time"]
            + [f"pos_{j}" for j in TARGET_POSE]
            + [f"force_{a}" for a in actuator_names]
            + [f"iaa_{j}_g" for j in TARGET_POSE]
            + [f"iaa_{j}_c" for j in TARGET_POSE]
            + [f"iaa_{j}_t" for j in TARGET_POSE]
            + [f"iaa_{j}_total" for j in TARGET_POSE]
        )

        camera_id = 0
        for i in range(physics.model.ncam):
            if physics.model.id2name(i, "camera") == "face_on":
                camera_id = i

        for i in range(steps):
            action = controller.get_action(physics)
            physics.set_control(action)
            physics.step()

            # Record
            pixels = physics.render(height=480, width=640, camera_id=camera_id)
            frames.append(pixels)

            # Compute IAA & CF
            iaa = iaa_helper.compute_induced_accelerations(physics)
            cf = iaa_helper.compute_counterfactuals(physics)
            mass_matrix = iaa_helper.get_mass_matrix(physics)

            # Emit Data Stream
            try:
                packet = {
                    "time": physics.data.time,
                    "qpos": physics.data.qpos,
                    "qvel": physics.data.qvel,
                    "qfrc_actuator": physics.data.qfrc_actuator,
                    "iaa": iaa,
                    "cf": cf,
                    "mass_matrix": mass_matrix,
                }
                print(f"DATA_JSON:{json.dumps(packet, default=np_encoder)}", flush=True)
            except Exception as e:
                # Avoid crashing loop on serialization error, just log
                print(f"DEBUG: Data serialization failed: {e}", flush=True)

            # Log Data (CSV)
            row = [physics.data.time]
            for j in TARGET_POSE:
                try:
                    val = physics.named.data.qpos[j]
                except Exception:
                    val = 0
                row.append(val)
            for a in actuator_names:
                try:
                    idx = actuators[a]
                    val = physics.data.actuator_force[idx]
                except Exception:
                    val = 0
                row.append(val)

            # Append IAA
            if iaa:
                for j in TARGET_POSE:
                    try:
                        # Find DOF address
                        j_id = physics.model.name2id(j, "joint")
                        dof_adr = physics.model.jnt_dofadr[j_id]

                        g_val = iaa["gravity"][dof_adr]
                        c_val = iaa["coriolis"][dof_adr]
                        t_val = iaa["control"][dof_adr]
                        tot_val = g_val + c_val + t_val

                        row.extend([g_val, c_val, t_val, tot_val])
                    except Exception:
                        row.extend([0, 0, 0, 0])
            else:
                # Fill zeros
                row.extend([0] * (4 * len(TARGET_POSE)))

            data_rows.append(row)

            if i % 30 == 0:
                print(f"Frame {i}/{steps}")

        # Save
        imageio.mimsave(output_video, frames, fps=fps)
        with open(output_data, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)

        if save_path:
            save_state(physics, save_path)


if __name__ == "__main__":
    run_simulation()
