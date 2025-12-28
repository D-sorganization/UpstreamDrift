class DrakeRecorder:
    """Records simulation data for analysis."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.times: list[float] = []
        self.q_history: list[np.ndarray] = []
        self.v_history: list[np.ndarray] = []
        self.club_head_pos_history: list[np.ndarray] = []
        # Store computed metrics
        self.induced_accelerations: dict[str, list[np.ndarray]] = {}
        self.counterfactuals: dict[str, list[np.ndarray]] = {}
        self.is_recording = False

    def start(self) -> None:
        self.reset()
        self.is_recording = True

    def stop(self) -> None:
        self.is_recording = False

    def record(
        self,
        t: float,
        q: np.ndarray,
        v: np.ndarray,
        club_pos: np.ndarray | None = None,
    ) -> None:
        if not self.is_recording:
            return
        self.times.append(t)
        self.q_history.append(q.copy())
        self.v_history.append(v.copy())
        if club_pos is not None:
            self.club_head_pos_history.append(club_pos.copy())
        else:
            self.club_head_pos_history.append(np.zeros(3))

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Implement RecorderInterface."""
        times = np.array(self.times)
        if field_name == "club_head_position":
            return times, np.array(self.club_head_pos_history)
        if field_name == "joint_positions":
            return times, np.array(self.q_history)
        if field_name == "joint_velocities":
            return times, np.array(self.v_history)

        # Fallback
        return times, []

    def get_induced_acceleration_series(self, source_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get induced accelerations."""
        if source_name not in self.induced_accelerations:
             return np.array([]), np.array([])

        times = np.array(self.times)
        # Ensure alignment
        vals = self.induced_accelerations[source_name]
        if len(vals) != len(times):
             # Truncate to match
             min_len = min(len(vals), len(times))
             return times[:min_len], np.array(vals[:min_len])

        return times, np.array(vals)

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get counterfactual data."""
        if cf_name not in self.counterfactuals:
             return np.array([]), np.array([])

        times = np.array(self.times)
        vals = self.counterfactuals[cf_name]

        if len(vals) != len(times):
             min_len = min(len(vals), len(times))
             return times[:min_len], np.array(vals[:min_len])

        return times, np.array(vals)
