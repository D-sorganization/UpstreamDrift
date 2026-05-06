function [J, terms] = compute_cost(theta, target, sim_fn, opts)
%COMPUTE_COST Scalar swing-matching cost (see COST_FUNCTION_SPEC.md).
%
%   [J, TERMS] = COMPUTE_COST(THETA, TARGET, SIM_FN, OPTS) evaluates
%
%       J(THETA) = w_p * mean_n( ||r_butt_sim - r_butt_meas||^2
%                              + ||r_ch_sim   - r_ch_meas||^2 )
%                + w_o * mean_n( d_geo(R_sim, R_meas)^2 )
%                + w_a * ||r_ch_sim(t_impact) - r_ch_meas(t_impact)||^2
%                + lambda * R(theta)
%
%   where R(theta) is one of total_work, peak_power, torque_l2, coeff_l2,
%   effort_l2, smoothness_l2 depending on OPTS.regularizer.
%
%   Inputs:
%     THETA  - real, finite coefficient vector (column or row).
%     TARGET - struct conforming to CLUB_IK_SPEC.md, with fields:
%                .time (Nx1), .butt (Nx3), .clubhead (Nx3),
%                .club_quat (Nx4 unit quaternion [w x y z]),
%                .impact_idx (1<=k<=N).
%     SIM_FN - function_handle: sim_out = SIM_FN(theta) returning a struct
%              with at least fields .butt (Nx3), .clubhead (Nx3),
%              .club_quat (Nx4), and (for "total_work" / "peak_power" /
%              "torque_l2" regularizers) .time, .tau, .omega.
%     OPTS   - struct from default_cost_options() with optional overrides.
%
%   Outputs:
%     J      - finite, non-negative scalar total cost.
%     TERMS  - struct with non-negative scalar fields:
%                .position .orientation .impact_anchor .regularizer .total
%              and TERMS.total == J within eps.
    arguments
        theta  (:,1) double {validators.mustBeFiniteVector}
        target (1,1) struct {validators.mustHaveFields(target, ...
            ["time", "butt", "clubhead", "club_quat", "impact_idx"])}
        sim_fn (1,1) function_handle
        opts   (1,1) struct = default_cost_options()
    end

    sim_out = sim_fn(theta);
    if ~isstruct(sim_out) || ~isscalar(sim_out)
        error("compute_cost:badSimOut", ...
              "sim_fn must return a scalar struct.");
    end
    validators.mustHaveFields(sim_out, ["butt", "clubhead", "club_quat"]);

    N = numel(target.time);
    local_check_traj_shape(target.butt,      N, 3, "target.butt");
    local_check_traj_shape(target.clubhead,  N, 3, "target.clubhead");
    local_check_traj_shape(target.club_quat, N, 4, "target.club_quat");
    local_check_traj_shape(sim_out.butt,     N, 3, "sim_out.butt");
    local_check_traj_shape(sim_out.clubhead, N, 3, "sim_out.clubhead");
    local_check_traj_shape(sim_out.club_quat, N, 4, "sim_out.club_quat");

    pos_term = local_position_term(sim_out, target);
    ori_term = local_orientation_term(sim_out, target);
    anc_term = local_anchor_term(sim_out, target);
    reg_term = local_regularizer_term(theta, sim_out, opts);

    terms = struct();
    terms.position      = opts.w_position      * pos_term;
    terms.orientation   = opts.w_orientation   * ori_term;
    terms.impact_anchor = opts.w_anchor_impact * anc_term;
    terms.regularizer   = opts.lambda          * reg_term;
    terms.total         = terms.position + terms.orientation ...
                        + terms.impact_anchor + terms.regularizer;

    J = terms.total;

    assert(isscalar(J) && isfinite(J) && J >= 0, ...
        "Postcondition: J must be a finite, non-negative scalar (got %g)", J);
    assert(abs(terms.total - J) <= eps(max(1, abs(J))) * 8, ...
        "Postcondition: terms.total must equal J within eps");
    assert(terms.position      >= 0 && terms.orientation >= 0 && ...
           terms.impact_anchor >= 0 && terms.regularizer >= 0, ...
        "Postcondition: every terms field must be non-negative");
end

% ---------- Term helpers (LOD <= 2; each is a single small operation) ----------

function val = local_position_term(sim_out, target)
    db = sim_out.butt     - target.butt;
    dc = sim_out.clubhead - target.clubhead;
    per_frame = sum(db .^ 2, 2) + sum(dc .^ 2, 2);
    val = mean(per_frame);
end

function val = local_orientation_term(sim_out, target)
    % Geodesic angle via quaternion dot, with abs() to handle q ~ -q.
    dots = sum(sim_out.club_quat .* target.club_quat, 2);
    dots = min(1, max(-1, abs(dots)));   % clamp for acos numerical safety
    angles = 2 * acos(dots);             % radians
    val = mean(angles .^ 2);
end

function val = local_anchor_term(sim_out, target)
    k = target.impact_idx;
    if ~(isnumeric(k) && isscalar(k) && k == floor(k) && k >= 1 && ...
         k <= size(target.clubhead, 1))
        error("compute_cost:badImpactIdx", ...
              "target.impact_idx must be a scalar integer in [1, N].");
    end
    d = sim_out.clubhead(k, :) - target.clubhead(k, :);
    val = sum(d .^ 2);
end

function val = local_regularizer_term(theta, sim_out, opts)
    switch lower(string(opts.regularizer))
        case "total_work"
            val = compute_total_work(sim_out);
        case "peak_power"
            validators.mustHaveFields(sim_out, ["tau", "omega"]);
            val = max(sum(abs(sim_out.tau .* sim_out.omega), 2));
        case "torque_l2"
            validators.mustHaveFields(sim_out, ["time", "tau"]);
            val = trapz(sim_out.time, sum(sim_out.tau .^ 2, 2));
        case "coeff_l2"
            val = sum(theta .^ 2);
        case "effort_l2"
            validators.mustHaveFields(sim_out, "tau");
            tau = sim_out.tau;
            tau_ref = local_resolve_tau_reference(opts, size(tau));
            w = local_resolve_reg_weights(opts, size(tau, 2));
            val = mean((tau - tau_ref) .^ 2 .* w, 'all');
        case "smoothness_l2"
            validators.mustHaveFields(sim_out, "tau");
            tau = sim_out.tau;
            if size(tau, 1) < 2
                val = 0;
            else
                w = local_resolve_reg_weights(opts, size(tau, 2));
                dtau = diff(tau, 1, 1);
                val = mean(dtau .^ 2 .* w, 'all');
            end
        otherwise
            error("compute_cost:badRegularizer", ...
                  "Unknown regularizer '%s'. Expected one of: " + ...
                  "total_work, peak_power, torque_l2, coeff_l2, " + ...
                  "effort_l2, smoothness_l2.", ...
                  string(opts.regularizer));
    end
    assert(isscalar(val) && isfinite(val) && val >= 0, ...
        "Postcondition: regularizer term must be finite and non-negative");
end

function tau_ref = local_resolve_tau_reference(opts, tau_size)
    if ~isfield(opts, "tau_reference") || isempty(opts.tau_reference)
        tau_ref = zeros(tau_size);
        return;
    end
    tau_ref = opts.tau_reference;
    if ~isnumeric(tau_ref) || ~isreal(tau_ref) || any(~isfinite(tau_ref(:)))
        error("compute_cost:badTauReference", ...
              "opts.tau_reference must be a real, finite numeric array.");
    end
    if isvector(tau_ref) && numel(tau_ref) == tau_size(2)
        tau_ref = repmat(tau_ref(:).', tau_size(1), 1);
    end
    if ~isequal(size(tau_ref), tau_size)
        error("compute_cost:badTauReferenceShape", ...
              "opts.tau_reference must broadcast to size [%d %d]; got [%s].", ...
              tau_size(1), tau_size(2), num2str(size(tau_ref)));
    end
end

function w = local_resolve_reg_weights(opts, n_joints)
    if ~isfield(opts, "regularizer_weights") || isempty(opts.regularizer_weights)
        w = ones(1, n_joints);
        return;
    end
    w = opts.regularizer_weights;
    if ~isnumeric(w) || ~isreal(w) || any(~isfinite(w(:))) || any(w(:) < 0)
        error("compute_cost:badRegWeights", ...
              "opts.regularizer_weights must be real, finite, non-negative.");
    end
    if numel(w) ~= n_joints
        error("compute_cost:badRegWeightsLen", ...
              "opts.regularizer_weights must have length %d; got %d.", ...
              n_joints, numel(w));
    end
    w = reshape(w, 1, n_joints);
end

function local_check_traj_shape(A, nrows, ncols, name)
    if ~isnumeric(A) || ~isreal(A) || ~ismatrix(A) || ...
       size(A, 1) ~= nrows || size(A, 2) ~= ncols
        error("compute_cost:badShape", ...
              "%s must be a real %dx%d matrix.", name, nrows, ncols);
    end
    if any(~isfinite(A(:)))
        error("compute_cost:notFinite", ...
              "%s contains NaN or Inf.", name);
    end
end
