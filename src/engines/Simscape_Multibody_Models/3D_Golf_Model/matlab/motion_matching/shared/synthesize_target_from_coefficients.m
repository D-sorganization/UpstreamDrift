function target = synthesize_target_from_coefficients(theta, opts)
%SYNTHESIZE_TARGET_FROM_COEFFICIENTS  TDD oracle for motion matching.
%   target = SYNTHESIZE_TARGET_FROM_COEFFICIENTS(THETA, OPTS) runs the
%   Simscape forward simulator with coefficient vector THETA and returns
%   a target struct conforming to CLUB_IK_SPEC.md.
%
%   This is the canonical oracle: any optimizer that cannot recover theta
%   (within RMSE < 1mm) when given synthesize_target_from_coefficients(theta)
%   as input is broken — not the data.
%
%   THETA must be a real, finite vector of length n_joints*7 ordered
%   [A B C D E F G] per joint.
%
%   OPTS is the result of DEFAULT_SYNTH_OPTIONS() with optional overrides.
%   Notably .sim_opts is merged on top of DEFAULT_SIM_OPTIONS() so the
%   wrapper SIMULATE_WITH_COEFFICIENTS performs the only Simscape call —
%   this function never invokes sim()/parsim() directly (DRY, see #018).
%
%   Output target struct (CLUB_IK_SPEC.md schema):
%     .time       (N,1) double  simulation timegrid (s), starts at 0
%     .butt       (N,3) double  butt position (m, world frame)
%     .clubhead   (N,3) double  clubhead position (m, world frame)
%     .club_quat  (N,4) double  unit quaternion [w x y z]
%     .impact_idx scalar uint32 argmax ||d r_clubhead/dt||
%     .source     struct        provenance (see below)
%
%   Provenance fields populated for synthetic source:
%     .filename     ""              (no file on disk)
%     .format       "synthetic"
%     .subject_id   from opts
%     .trial_id     from opts
%     .sha256       sha256(theta bytes)  — used as theta_hash
%     .theta_truth  the input theta (so the oracle can be recovered)
%     .git_commit   current git HEAD or "unknown"
%
%   Preconditions (arguments + asserts):
%     - THETA is real, finite, length n_joints*7.
%     - THETA lies within bounds from generateRandomCoefficients
%       (A,B in +/-1000; C,D in +/-500; E,F in +/-100; G in +/-25).
%     - opts.sample_rate > 0 and opts.simulation_time in (0, 1].
%
%   Postconditions:  every CLUB_IK_SPEC.md §"Validation rules" item.
%
%   GitHub issue: #014 / #3983.
%   Depends on: #018 (simulate_with_coefficients).
%
%   See also: SIMULATE_WITH_COEFFICIENTS, DEFAULT_SYNTH_OPTIONS,
%             DEFAULT_SIM_OPTIONS.

    arguments
        theta (:,1) double {mustBeReal, mustBeFinite}
        opts  (1,1) struct = default_synth_options()
    end

    % --- 1. Validate opts and theta bounds ---------------------------------
    opts = local_fill_defaults(opts);
    local_validate_theta_bounds(theta);

    % --- 2. Build sim options merging defaults + caller overrides ----------
    sim_opts = default_sim_options();
    sim_opts.simulation_time = double(opts.simulation_time);
    sim_opts.sample_rate     = double(opts.sample_rate);
    sim_opts.verbosity       = "Silent";
    if isstruct(opts.sim_opts)
        f = fieldnames(opts.sim_opts);
        for k = 1:numel(f)
            sim_opts.(f{k}) = opts.sim_opts.(f{k});
        end
    end

    % --- 3. Run the canonical Simscape wrapper -----------------------------
    sim_out = simulate_with_coefficients(theta, sim_opts);
    if sim_out.solver_status == "failed"
        error("synthesize_target_from_coefficients:simFailed", ...
              "Underlying simulate_with_coefficients failed: %s", ...
              sim_out.status_message);
    end

    % --- 4. Convert sim_out -> target schema -------------------------------
    butt     = double(sim_out.r_butt);
    clubhead = double(sim_out.r_clubhead);
    quat     = local_normalise_quat_rows(double(sim_out.q_club));
    time     = double(sim_out.time(:));

    if opts.add_noise
        rng_state = rng();             %#ok<NASGU> % preserved for caller
        rng(0, 'twister');             % deterministic noise per call
        sigma = double(opts.noise_sigma_m);
        butt     = butt     + sigma * randn(size(butt));
        clubhead = clubhead + sigma * randn(size(clubhead));
    end

    impact_idx = uint32(detect_clubhead_impact(time, clubhead));

    % --- 5. Provenance ------------------------------------------------------
    theta_bytes = typecast(double(theta(:)), 'uint8');
    theta_hash  = sha256_of_bytes(theta_bytes(:));

    source = struct( ...
        "filename",    "", ...
        "format",      "synthetic", ...
        "subject_id",  string(opts.subject_id), ...
        "trial_id",    string(opts.trial_id), ...
        "sha256",      theta_hash, ...
        "theta_truth", double(theta(:)), ...
        "git_commit",  local_safe_git_commit());

    target = struct( ...
        "time",       time, ...
        "butt",       butt, ...
        "clubhead",   clubhead, ...
        "club_quat",  quat, ...
        "impact_idx", impact_idx, ...
        "source",     source);

    % --- 6. Postconditions: CLUB_IK_SPEC.md §"Validation rules" ------------
    N = numel(target.time);
    assert(N >= 2, ...
        "synthesize_target_from_coefficients:tooFewSamples", ...
        "Postcondition: at least 2 samples on simulation grid");
    assert(all(diff(target.time) > 0), ...
        "synthesize_target_from_coefficients:timeNotIncreasing", ...
        "Postcondition: time strictly increasing");
    assert(abs(target.time(1)) < 1e-9, ...
        "synthesize_target_from_coefficients:timeStart", ...
        "Postcondition: time(1) must be 0");
    assert(target.time(end) <= double(opts.simulation_time) + eps(1), ...
        "synthesize_target_from_coefficients:timeEnd", ...
        "Postcondition: time(end) <= simulation_time + eps");
    assert(size(target.butt,1) == N && size(target.clubhead,1) == N && ...
           size(target.club_quat,1) == N, ...
        "synthesize_target_from_coefficients:rowMismatch", ...
        "Postcondition: trajectory rows must equal numel(time)");
    assert(all(isfinite(target.butt(:))) && all(isfinite(target.clubhead(:))), ...
        "synthesize_target_from_coefficients:nonFinitePos", ...
        "Postcondition: positions must be finite");
    assert(all(vecnorm(target.butt,2,2) < 5) && ...
           all(vecnorm(target.clubhead,2,2) < 5), ...
        "synthesize_target_from_coefficients:posMagnitude", ...
        "Postcondition: ||r|| must be < 5 m");
    qn = sqrt(sum(target.club_quat.^2, 2));
    assert(all(abs(qn - 1) < 1e-6), ...
        "synthesize_target_from_coefficients:quatNotUnit", ...
        "Postcondition: club_quat rows must be unit-norm to 1e-6");
    assert(target.impact_idx >= 1 && target.impact_idx <= N, ...
        "synthesize_target_from_coefficients:impactBounds", ...
        "Postcondition: 1 <= impact_idx <= N");
    assert(strlength(target.source.sha256) == 64, ...
        "synthesize_target_from_coefficients:badHash", ...
        "Postcondition: source.sha256 must be 64 hex chars");
    assert(target.source.format == "synthetic", ...
        "synthesize_target_from_coefficients:badFormat", ...
        "Postcondition: source.format must be ""synthetic""");
    assert(isequal(target.source.theta_truth(:), double(theta(:))), ...
        "synthesize_target_from_coefficients:thetaTruthMismatch", ...
        "Postcondition: source.theta_truth must equal input theta");
end


%% =====================================================================
function opts = local_fill_defaults(opts)
%LOCAL_FILL_DEFAULTS  Backfill any missing fields from default_synth_options.
    defaults = default_synth_options();
    f = fieldnames(defaults);
    for k = 1:numel(f)
        if ~isfield(opts, f{k})
            opts.(f{k}) = defaults.(f{k});
        end
    end
    if ~(isnumeric(opts.sample_rate) && isscalar(opts.sample_rate) && opts.sample_rate > 0)
        error("synthesize_target_from_coefficients:badSampleRate", ...
              "Precondition: opts.sample_rate must be a positive scalar");
    end
    if ~(isnumeric(opts.simulation_time) && isscalar(opts.simulation_time) && ...
            opts.simulation_time > 0 && opts.simulation_time <= 1)
        error("synthesize_target_from_coefficients:badSimTime", ...
              "Precondition: opts.simulation_time must be in (0, 1]");
    end
end


%% =====================================================================
function local_validate_theta_bounds(theta)
%LOCAL_VALIDATE_THETA_BOUNDS  Enforce per-coefficient bounds matching
%   generateRandomCoefficients.m: |A|,|B| <= 1000; |C|,|D| <= 500;
%   |E|,|F| <= 100; |G| <= 25.
    if mod(numel(theta), 7) ~= 0
        error("synthesize_target_from_coefficients:badThetaLength", ...
              "Precondition: theta length must be a multiple of 7 (got %d)", ...
              numel(theta));
    end
    n_joints = numel(theta) / 7;
    M = reshape(theta, 7, n_joints).';   % rows = joints, cols = [A B C D E F G]
    bounds = [1000, 1000, 500, 500, 100, 100, 25];
    for c = 1:7
        if any(abs(M(:, c)) > bounds(c) + 1e-9)
            letter = char('A' + c - 1);
            error("synthesize_target_from_coefficients:thetaOutOfBounds", ...
                  "Precondition: coefficient %s exceeds +/-%g", ...
                  letter, bounds(c));
        end
    end
end


%% =====================================================================
function q = local_normalise_quat_rows(q)
%LOCAL_NORMALISE_QUAT_ROWS  Force unit-norm and q(:,1) >= 0 sign convention.
    if isempty(q) || all(isnan(q(:)))
        % Fall back to identity quaternion repeated; this should not happen
        % on a successful sim but guards the postcondition.
        N = max(size(q, 1), 1);
        q = repmat([1, 0, 0, 0], N, 1);
        return;
    end
    nrm = sqrt(sum(q.^2, 2));
    nrm(nrm == 0) = 1;
    q = q ./ nrm;
    flip = q(:, 1) < 0;
    q(flip, :) = -q(flip, :);
end


%% =====================================================================
function s = local_safe_git_commit()
    s = "unknown";
    try
        [status, out] = system('git rev-parse HEAD');
        if status == 0
            s = string(strtrim(out));
        end
    catch
    end
end
