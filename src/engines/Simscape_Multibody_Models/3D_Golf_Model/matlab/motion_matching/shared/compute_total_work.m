function W = compute_total_work(sim_out)
%COMPUTE_TOTAL_WORK Total mechanical work integral across all joints.
%
%   W = COMPUTE_TOTAL_WORK(SIM_OUT) returns the scalar
%       W = integral over t of sum_j |tau_j(t) * omega_j(t)| dt
%   computed by trapezoidal rule on SIM_OUT.time.
%
%   SIM_OUT must be a struct with fields:
%     .time  (N x 1 double, monotonic seconds)
%     .tau   (N x n_joints double, joint torques in N*m)
%     .omega (N x n_joints double, joint angular velocities in rad/s)
%
%   Both eccentric (tau and omega opposite sign) and concentric work are
%   counted positively via the elementwise abs. See COST_FUNCTION_SPEC.md
%   § Regularizer.
%
%   Postcondition: W >= 0 and finite.
    arguments
        sim_out (1,1) struct {validators.mustHaveFields( ...
            sim_out, ["time", "tau", "omega"])}
    end

    t   = sim_out.time(:);
    tau = sim_out.tau;
    om  = sim_out.omega;

    if size(tau, 1) ~= numel(t) || size(om, 1) ~= numel(t)
        error("compute_total_work:shapeMismatch", ...
              "tau and omega must have N rows matching length(time)=%d.", numel(t));
    end
    if ~isequal(size(tau), size(om))
        error("compute_total_work:shapeMismatch", ...
              "tau and omega must have the same shape.");
    end

    integrand = sum(abs(tau .* om), 2);  % N x 1
    W = trapz(t, integrand);

    assert(isfinite(W), "Postcondition: total work must be finite");
    assert(W >= 0,      "Postcondition: total work must be non-negative");
end
