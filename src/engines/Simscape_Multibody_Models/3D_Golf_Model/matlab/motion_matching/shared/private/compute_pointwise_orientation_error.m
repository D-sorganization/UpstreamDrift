function err_deg = compute_pointwise_orientation_error(sim_quat, meas_quat)
%COMPUTE_POINTWISE_ORIENTATION_ERROR  Per-frame geodesic angle (degrees).
%
%   ERR_DEG = COMPUTE_POINTWISE_ORIENTATION_ERROR(SIM_QUAT, MEAS_QUAT)
%   computes the geodesic angle between two (N,4) unit-quaternion
%   trajectories, returned as an (N,1) column vector in degrees.
%
%   Implementation:  angle = 2 * acos(|q1 . q2|), then converted to degrees.
%   The abs() handles the q ~ -q double-cover sign ambiguity (per
%   COST_FUNCTION_SPEC.md, "Numerical considerations").
%
%   Preconditions:
%     - SIM_QUAT and MEAS_QUAT are real (N,4) matrices of the same size.
%   Postconditions:
%     - ERR_DEG is (N,1) double in [0, 180], finite where inputs are finite.
    arguments
        sim_quat  (:,4) double {mustBeReal}
        meas_quat (:,4) double {mustBeReal}
    end
    if size(sim_quat, 1) ~= size(meas_quat, 1)
        error("compute_pointwise_orientation_error:sizeMismatch", ...
              "sim_quat and meas_quat must have the same number of rows.");
    end
    dots = sum(sim_quat .* meas_quat, 2);
    dots = min(1, max(-1, abs(dots)));   % clamp for acos numerical safety
    angles_rad = 2 * acos(dots);
    err_deg = angles_rad * (180 / pi);
    assert(all((err_deg >= -1e-9 & err_deg <= 180 + 1e-9) | isnan(err_deg)), ...
        "Postcondition: orientation error must lie in [0, 180] deg");
end
