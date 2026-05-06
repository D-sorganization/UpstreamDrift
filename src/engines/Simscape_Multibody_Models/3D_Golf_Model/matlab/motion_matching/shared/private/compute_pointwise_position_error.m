function err = compute_pointwise_position_error(sim_xyz, meas_xyz)
%COMPUTE_POINTWISE_POSITION_ERROR  Per-frame Euclidean position error (metres).
%
%   ERR = COMPUTE_POINTWISE_POSITION_ERROR(SIM_XYZ, MEAS_XYZ) returns an
%   (N,1) column vector of non-negative Euclidean distances between
%   simulated and measured (N,3) position trajectories.
%
%   Preconditions:
%     - SIM_XYZ and MEAS_XYZ are real (N,3) matrices of the same size.
%   Postconditions:
%     - ERR is (N,1) double, non-negative, finite where inputs are finite.
    arguments
        sim_xyz  (:,3) double {mustBeReal}
        meas_xyz (:,3) double {mustBeReal}
    end
    if size(sim_xyz, 1) ~= size(meas_xyz, 1)
        error("compute_pointwise_position_error:sizeMismatch", ...
              "sim_xyz and meas_xyz must have the same number of rows.");
    end
    d = sim_xyz - meas_xyz;
    err = sqrt(sum(d .* d, 2));
    assert(all(err >= 0 | isnan(err)), ...
        "Postcondition: pointwise position error must be non-negative");
end
