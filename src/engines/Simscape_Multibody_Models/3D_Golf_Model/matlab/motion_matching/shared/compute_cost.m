function [J, terms] = compute_cost(theta, target, sim_fn, opts)
%COMPUTE_COST Scalar swing-matching cost (see COST_FUNCTION_SPEC.md).
%
%   [J, TERMS] = COMPUTE_COST(THETA, TARGET, SIM_FN, OPTS) evaluates
%
%       J(THETA) = w_pg * mean_n( ||r_grip_sim - r_grip_meas||^2 )
%                + w_pc * mean_n( ||r_ch_sim   - r_ch_meas||^2 )      (default 0)
%                + w_og * mean_n( d_geo(R_grip_sim, R_grip_meas)^2 )
%                + w_oc * mean_n( d_geo(R_club_sim, R_club_meas)^2 )  (default 0)
%                + w_a * ||r_grip_sim(t_impact) - r_grip_meas(t_impact)||^2
%                + lambda * R(theta)
%
%   where R(theta) is one of total_work, peak_power, torque_l2, coeff_l2,
%   effort_l2, smoothness_l2 depending on OPTS.regularizer.
%
%   The **grip / mid-hands** position is the primary motion-matching
%   anchor because it is the rigid contact point between the body
%   kinematics and the club.  Clubhead and club-orientation terms are
%   available as low-weight secondary signals (default 0) so the cost
%   doesn't penalise the player having a different physical club length
%   or shaft flex than the modeled club — those are NOT what we are
%   trying to learn.
%
%   Backward compatibility: if the new w_position_* / w_orientation_*
%   fields are absent, the old w_position / w_orientation fields are
%   used and the cost reverts to the legacy butt+clubhead formulation.
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

    % Resolve grip aliases — older callers populate butt; newer ones grip.
    target  = local_resolve_grip_aliases(target);
    sim_out = local_resolve_grip_aliases(sim_out);

    N = numel(target.time);
    local_check_traj_shape(target.grip,      N, 3, "target.grip");
    local_check_traj_shape(target.clubhead,  N, 3, "target.clubhead");
    local_check_traj_shape(target.club_quat, N, 4, "target.club_quat");
    local_check_traj_shape(sim_out.grip,     N, 3, "sim_out.grip");
    local_check_traj_shape(sim_out.clubhead, N, 3, "sim_out.clubhead");
    local_check_traj_shape(sim_out.club_quat, N, 4, "sim_out.club_quat");

    [w_pg, w_pc, w_og, w_oc] = local_resolve_weights(opts);

    pos_grip = local_pos_term(sim_out.grip,     target.grip);
    pos_club = local_pos_term(sim_out.clubhead, target.clubhead);
    ori_grip = local_orient_term(sim_out, target, "grip");
    ori_club = local_orient_term(sim_out, target, "club");
    anc_term = local_anchor_term(sim_out, target);
    reg_term = local_regularizer_term(theta, sim_out, opts);

    terms = struct();
    terms.position_grip     = w_pg              * pos_grip;
    terms.position_clubhead = w_pc              * pos_club;
    terms.orientation_grip  = w_og              * ori_grip;
    terms.orientation_club  = w_oc              * ori_club;
    terms.impact_anchor     = opts.w_anchor_impact * anc_term;
    terms.regularizer       = opts.lambda          * reg_term;
    % Legacy aggregate names retained for older callers / dashboards.
    terms.position      = terms.position_grip + terms.position_clubhead;
    terms.orientation   = terms.orientation_grip + terms.orientation_club;
    terms.total         = terms.position_grip + terms.position_clubhead ...
                        + terms.orientation_grip + terms.orientation_club ...
                        + terms.impact_anchor + terms.regularizer;

    J = terms.total;

    assert(isscalar(J) && isfinite(J) && J >= 0, ...
        "Postcondition: J must be a finite, non-negative scalar (got %g)", J);
    assert(abs(terms.total - J) <= eps(max(1, abs(J))) * 8, ...
        "Postcondition: terms.total must equal J within eps");
    assert(terms.position_grip     >= 0 && terms.position_clubhead >= 0 && ...
           terms.orientation_grip  >= 0 && terms.orientation_club  >= 0 && ...
           terms.impact_anchor     >= 0 && terms.regularizer       >= 0, ...
        "Postcondition: every terms field must be non-negative");
end

% ---------- Term helpers (LOD <= 2; each is a single small operation) ----------

function val = local_pos_term(sim_pts, target_pts)
    d = sim_pts - target_pts;
    val = mean(sum(d .^ 2, 2));
end

function val = local_orient_term(sim_out, target, which)
%LOCAL_ORIENT_TERM  Mean squared geodesic angle between sim/target quats.
%   WHICH = "grip" or "club".  When the requested quaternion field is
%   missing on either side, the term gracefully returns 0.
    sim_field    = sprintf('%s_quat', which);
    target_field = sprintf('%s_quat', which);
    if ~isfield(sim_out, sim_field) || ~isfield(target, target_field)
        val = 0; return;
    end
    qs = sim_out.(sim_field);
    qt = target.(target_field);
    if isempty(qs) || isempty(qt) || size(qs, 1) ~= size(qt, 1)
        val = 0; return;
    end
    dots = sum(qs .* qt, 2);
    dots = min(1, max(-1, abs(dots)));
    angles = 2 * acos(dots);
    val = mean(angles .^ 2);
end

function val = local_anchor_term(sim_out, target)
%LOCAL_ANCHOR_TERM  Hard impact-frame anchor on the GRIP position.
%   Anchoring on the grip is the right physics: the body delivers the
%   grip to the right place at the right time; the clubhead is along
%   for the rigid-extension ride.  (Legacy callers can still raise
%   w_position_clubhead to add a clubhead penalty.)
    k = target.impact_idx;
    if ~(isnumeric(k) && isscalar(k) && k == floor(k) && k >= 1 && ...
         k <= size(target.grip, 1))
        error("compute_cost:badImpactIdx", ...
              "target.impact_idx must be a scalar integer in [1, N].");
    end
    d = sim_out.grip(k, :) - target.grip(k, :);
    val = sum(d .^ 2);
end

function s = local_resolve_grip_aliases(s)
%LOCAL_RESOLVE_GRIP_ALIASES  Fill in `grip` from `butt` (or vice-versa).
%   The two are synonyms in this codebase — the historical name was
%   `butt`, but the physical meaning is mid-hands position on the shaft.
    if ~isfield(s, 'grip') && isfield(s, 'butt')
        s.grip = s.butt;
    elseif ~isfield(s, 'butt') && isfield(s, 'grip')
        s.butt = s.grip;
    end
end

function [w_pg, w_pc, w_og, w_oc] = local_resolve_weights(opts)
%LOCAL_RESOLVE_WEIGHTS  Map opts to grip/clubhead position+orientation weights.
%   New callers set w_position_grip / w_position_clubhead /
%   w_orientation_grip / w_orientation_club.  Legacy callers populate
%   w_position / w_orientation only — in that case we put the entire
%   position weight on grip (matching the old behaviour where grip and
%   clubhead were summed equally) and the orientation weight on the
%   club (the only orientation we historically tracked).
    if isfield(opts, 'w_position_grip')
        w_pg = double(opts.w_position_grip);
    elseif isfield(opts, 'w_position')
        w_pg = double(opts.w_position);
    else
        w_pg = 1.0;
    end
    if isfield(opts, 'w_position_clubhead')
        w_pc = double(opts.w_position_clubhead);
    elseif isfield(opts, 'w_position')
        w_pc = double(opts.w_position);   % legacy: butt+clubhead summed
    else
        w_pc = 0.0;
    end
    if isfield(opts, 'w_orientation_grip')
        w_og = double(opts.w_orientation_grip);
    else
        w_og = 0.0;
    end
    if isfield(opts, 'w_orientation_club')
        w_oc = double(opts.w_orientation_club);
    elseif isfield(opts, 'w_orientation')
        w_oc = double(opts.w_orientation);
    else
        w_oc = 0.0;
    end
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
