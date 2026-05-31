// AUTO-GENERATED from configs/ux/field_metadata.yaml — DO NOT EDIT BY HAND.
// Regenerate with: python3 scripts/ux/generate_field_metadata_ts.py
// The YAML is the single source of truth (epic #5968, DRY).

export interface FieldMetadata {
  id: string;
  label: string;
  shortHelp: string;
  longHelp: string;
  units: string | null;
  validRange: [number, number] | string[] | null;
  default: unknown;
  defaultSource: string;
  consumers: string[];
  producers: string[];
  example: string;
}

export const FIELD_METADATA: FieldMetadata[] = [
  {
    "id": "actuator.control_type",
    "label": "Control type",
    "shortHelp": "How this actuator's command is produced.",
    "longHelp": "- `constant`: fixed torque/force throughout the run.\n- `polynomial`: time-parameterised polynomial (good for scripted swings).\n- `pd_gains`: closed-loop proportional-derivative around a target.\n- `trajectory`: replay a recorded torque/position trace.\n",
    "units": null,
    "validRange": [
      "constant",
      "polynomial",
      "pd_gains",
      "trajectory"
    ],
    "default": "constant",
    "defaultSource": "Simplest control mode; safe across all engines.",
    "consumers": [],
    "producers": [],
    "example": "constant"
  },
  {
    "id": "actuator.max_value",
    "label": "Maximum command",
    "shortHelp": "Upper bound of the actuator's command range.",
    "longHelp": "Engine-reported upper limit for this actuator's command, taken from\nthe model's actuator definition.  ActuatorPanel uses it as the right\nend of each slider.  Read-only in the UI; it comes from the loaded\nmodel, not user input.\n",
    "units": null,
    "validRange": null,
    "default": 1.0,
    "defaultSource": "Model actuator ctrlrange upper bound (engine capabilities query).",
    "consumers": [
      "actuator.value"
    ],
    "producers": [],
    "example": "1.0"
  },
  {
    "id": "actuator.min_value",
    "label": "Minimum command",
    "shortHelp": "Lower bound of the actuator's command range.",
    "longHelp": "Engine-reported lower limit for this actuator's command, taken from\nthe model's actuator definition.  ActuatorPanel uses it as the left\nend of each slider.  Read-only in the UI; it comes from the loaded\nmodel, not user input.\n",
    "units": null,
    "validRange": null,
    "default": -1.0,
    "defaultSource": "Model actuator ctrlrange lower bound (engine capabilities query).",
    "consumers": [
      "actuator.value"
    ],
    "producers": [],
    "example": "-1.0"
  },
  {
    "id": "actuator.polynomial_coefficients",
    "label": "Polynomial coefficients",
    "shortHelp": "Coefficients of the time-parameterised command polynomial.",
    "longHelp": "Ordered coefficients `[c0, c1, c2, ...]` for the polynomial\n`c0 + c1*t + c2*t^2 + ...` evaluated at simulation time `t`.\nOnly used when `actuator.control_type` is `polynomial`.  The\nPolynomial Generator panel writes these.\n",
    "units": null,
    "validRange": null,
    "default": "[0.0]",
    "defaultSource": "Constant-zero polynomial matches the neutral default command.",
    "consumers": [],
    "producers": [
      "actuator.control_type"
    ],
    "example": "[0.0, 1.5, -0.25]"
  },
  {
    "id": "actuator.value",
    "label": "Command value",
    "shortHelp": "Commanded torque or force for this actuator.",
    "longHelp": "The control signal sent to the actuator each step.  Interpreted in\nthe actuator's own `units` (N*m for torque joints, N for linear\nactuators).  Must lie within [`actuator.min_value`,\n`actuator.max_value`]; values outside the range are clamped by the\nengine and flagged by preflight.\n",
    "units": null,
    "validRange": null,
    "default": 0.0,
    "defaultSource": "Zero command is the safe neutral start for every control type.",
    "consumers": [],
    "producers": [
      "actuator.min_value",
      "actuator.max_value"
    ],
    "example": "0.0"
  },
  {
    "id": "pose_studio.show_radians",
    "label": "Show in radians",
    "shortHelp": "Display joint angles in radians instead of degrees.",
    "longHelp": "The canonical convention is degrees; toggling this only changes how the\nspinboxes render and accept input.  Slider ticks always remain in tenths of a\ndegree (see src/tools/pose_studio/widgets/joint_panel.py).\n",
    "units": null,
    "validRange": null,
    "default": false,
    "defaultSource": "Degrees match the canonical pose interchange (REFERENCE_GOLFER_FIELDS).",
    "consumers": [],
    "producers": [],
    "example": "false"
  },
  {
    "id": "simulation.duration",
    "label": "Duration",
    "shortHelp": "How long the simulation runs (wall-clock seconds of sim time).",
    "longHelp": "Total simulated time, not real-time wall-clock duration.\nEngine adapters convert this to a step count using `timestep`.\nLonger durations linearly increase run time.\n",
    "units": "s",
    "validRange": [
      0.1,
      60.0
    ],
    "default": 3.0,
    "defaultSource": "Median golf-swing analysis window across MuJoCo benchmark suite (issue",
    "consumers": [],
    "producers": [],
    "example": "3.0"
  },
  {
    "id": "simulation.engine",
    "label": "Physics engine",
    "shortHelp": "Which physics backend executes the simulation.",
    "longHelp": "Each engine has different strengths and unit conventions.  Switching engines\nreloads ENGINE_DEFAULTS for duration and timestep (see ParameterPanel.tsx).\nPose data is converted to engine-native conventions by the canonical\npose-interchange layer (see src/shared/python/pose_interchange/).\n",
    "units": null,
    "validRange": [
      "mujoco",
      "drake",
      "pinocchio",
      "opensim",
      "myosim",
      "myosuite",
      "jaxsim",
      "pendulum_stub"
    ],
    "default": "mujoco",
    "defaultSource": "MuJoCo selected as default — fastest for interactive iteration (issue",
    "consumers": [],
    "producers": [],
    "example": "mujoco"
  },
  {
    "id": "simulation.gpu_acceleration",
    "label": "GPU acceleration",
    "shortHelp": "Use GPU when the engine supports it.",
    "longHelp": "Only relevant for MyoSim, IsaacGym, and Drake's GPU contact solver.\nIgnored by Pinocchio, OpenSim, and the pendulum stub.  Enabling without a CUDA\ndevice falls back to CPU silently — preflight will warn instead.\n",
    "units": null,
    "validRange": null,
    "default": false,
    "defaultSource": "Default off so the same config runs on laptops without CUDA (issue",
    "consumers": [],
    "producers": [],
    "example": "false"
  },
  {
    "id": "simulation.live_analysis",
    "label": "Live analysis",
    "shortHelp": "Stream analytics during the run instead of post-hoc.",
    "longHelp": "When on, the API streams per-step metrics over the WebSocket so plots update live.\nAdds 5-15% overhead.  Off is faster for batch runs that only need the final result.\n",
    "units": null,
    "validRange": null,
    "default": true,
    "defaultSource": "Default on for interactive use; Cross-Engine Dashboard turns this off for batch runs (issue",
    "consumers": [],
    "producers": [],
    "example": "true"
  },
  {
    "id": "simulation.model",
    "label": "Model",
    "shortHelp": "Which musculoskeletal/robot model the simulation loads.",
    "longHelp": "Identifier of the model the engine instantiates (URDF/MJCF/OSIM name\nor canonical model key).  Determines the available actuators and\njoints shown in ActuatorPanel.  Switching models reloads the\nactuator list.\n",
    "units": null,
    "validRange": null,
    "default": "",
    "defaultSource": "Empty means \"use the engine's bundled default model\" (ParameterPanel).",
    "consumers": [],
    "producers": [],
    "example": "humanoid"
  },
  {
    "id": "simulation.timestep",
    "label": "Timestep",
    "shortHelp": "Integrator step size in seconds.  Smaller = more accurate, slower.",
    "longHelp": "Engine integrators advance the simulation by this many seconds at a time.\nDrake recommends 1e-3 for contact-rich scenes; MuJoCo's default for floating-base\nhumanoids is 2e-3.  Setting this above the engine-recommended range will trigger\na non-blocking \"unusual value\" warning.\n",
    "units": "s",
    "validRange": [
      1e-06,
      1.0
    ],
    "default": 0.002,
    "defaultSource": "MuJoCo recommended timestep for floating-base humanoid (mujoco docs, 2024).",
    "consumers": [],
    "producers": [],
    "example": "0.002"
  }
];

export const FIELD_METADATA_BY_ID: Record<string, FieldMetadata> =
  Object.fromEntries(FIELD_METADATA.map((f) => [f.id, f]));

export function getFieldMetadata(id: string): FieldMetadata {
  const fm = FIELD_METADATA_BY_ID[id];
  if (fm === undefined) {
    throw new Error(`unknown field id: ${id}`);
  }
  return fm;
}
