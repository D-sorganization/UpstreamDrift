function aligned = align_to_simulation_grid(raw, opts)
%ALIGN_TO_SIMULATION_GRID  Detect impact, window, and resample a raw swing.
%
%   ALIGNED = ALIGN_TO_SIMULATION_GRID(RAW, OPTS) implements the time-alignment
%   policy from CLUB_IK_SPEC.md:
%     1. Detect impact = argmax of ||d r_clubhead/dt|| via 5-point central
%        differences over the raw mocap.
%     2. Define window [t_impact - opts.pre_impact_s, t_impact + opts.post_impact_s].
%     3. Resample (linear in position, SLERP in orientation) onto the simulation
%        timegrid 0:1/opts.sample_rate:T.
%     4. Anchor: re-time so measured impact lines up with opts.expected_impact_s.
%
%   RAW must be a struct with fields:
%     - time     (M,1) double  raw timestamps in seconds (monotonic)
%     - butt     (M,3) double  butt position (metres, world frame)
%     - clubhead (M,3) double  clubhead position (metres, world frame)
%     - club_quat (M,4) double unit quaternions [w x y z]
%
%   ALIGNED has the same fields plus a scalar uint32 IMPACT_IDX, all sampled on
%   the simulation grid.
%
%   Preconditions enforced via arguments block.  Postconditions:
%     - aligned.time strictly increasing, time(1) == 0.
%     - All trajectory arrays have N rows where N = numel(aligned.time).
%     - aligned.club_quat rows unit-norm to 1e-6.
%     - 1 <= aligned.impact_idx <= N.
    arguments
        raw (1,1) struct
        opts (1,1) struct = default_align_options()
    end

    required_fields = ["time", "butt", "clubhead", "club_quat"];
    missing = setdiff(required_fields, string(fieldnames(raw)));
    if ~isempty(missing)
        error("align_to_simulation_grid:missingField", ...
              "raw struct missing fields: %s", strjoin(missing, ", "));
    end

    t_raw = raw.time(:);
    M = numel(t_raw);
    if M < 5
        error("align_to_simulation_grid:tooFewFrames", ...
              "Need at least 5 frames for 5-point central differences (got %d)", M);
    end
    if any(diff(t_raw) <= 0)
        error("align_to_simulation_grid:nonMonotonicTime", ...
              "raw.time must be strictly increasing");
    end

    % --- Step 1: impact detection via 5-point central differences ---
    %     (DRY: shared with synthesize_target_from_coefficients.m).  If
    %     the caller has supplied a known-good impact time via
    %     opts.known_impact_s — e.g. read from the row-1 header of a
    %     Wiffle xlsx file — we use that directly, since the documented
    %     value is authoritative and the speed-argmax heuristic can
    %     latch onto the wrong local maximum on noisy traces.
    if isfield(opts, "known_impact_s") && ~isempty(opts.known_impact_s) && ...
            isfinite(double(opts.known_impact_s))
        t_impact_raw = double(opts.known_impact_s);
        if t_impact_raw < t_raw(1) || t_impact_raw > t_raw(end)
            error("align_to_simulation_grid:knownImpactOutOfRange", ...
                  "opts.known_impact_s=%g is outside raw t=[%g, %g]", ...
                  t_impact_raw, t_raw(1), t_raw(end));
        end
    else
        raw_impact_idx = detect_clubhead_impact(t_raw, raw.clubhead);
        t_impact_raw = t_raw(raw_impact_idx);
    end

    % --- Step 2: define window in raw time ---
    pre  = opts.pre_impact_s;
    post = opts.post_impact_s;
    window_start = max(t_raw(1), t_impact_raw - pre);
    window_end   = min(t_raw(end), t_impact_raw + post);

    % --- Step 4 first: anchor — re-time so that simulation impact lands at
    %     opts.expected_impact_s. Define the simulation time vector starting at 0.
    %     The mapping from sim_time -> raw_time is:
    %         raw_time = sim_time - opts.expected_impact_s + t_impact_raw
    expected = opts.expected_impact_s;
    sim_T = (window_end - window_start);
    sim_time = (0:1/opts.sample_rate:sim_T).';
    if isempty(sim_time) || sim_time(end) < sim_T - eps
        sim_time = [sim_time; sim_T];
    end

    sim_t_impact = expected;
    raw_query = sim_time - sim_t_impact + t_impact_raw;
    % Clamp to within [window_start, window_end]
    raw_query = max(window_start, min(window_end, raw_query));

    % --- Step 3: resample positions (linear) and orientation (SLERP) ---
    butt_q     = interp1(t_raw, raw.butt,     raw_query, "linear");
    clubhead_q = interp1(t_raw, raw.clubhead, raw_query, "linear");
    quat_q     = local_slerp_resample(t_raw, raw.club_quat, raw_query);
    if isfield(raw, "grip");      grip_q      = interp1(t_raw, raw.grip, raw_query, "linear"); else; grip_q = butt_q; end
    if isfield(raw, "grip_quat"); grip_quat_q = local_slerp_resample(t_raw, raw.grip_quat, raw_query); else; grip_quat_q = quat_q; end

    % Recompute impact on the simulation grid (should be near sim_t_impact)
    sim_impact_idx = detect_clubhead_impact(sim_time, clubhead_q);

    aligned = struct( ...
        "time",       sim_time, ...
        "grip",       grip_q, ...
        "grip_quat",  grip_quat_q, ...
        "butt",       butt_q, ...           % alias of grip for backward compat
        "clubhead",   clubhead_q, ...
        "club_quat",  quat_q, ...
        "impact_idx", uint32(sim_impact_idx));

    % --- Postconditions per CLUB_IK_SPEC.md §"Validation rules" ---
    N = numel(aligned.time);
    assert(N >= 2, "Postcondition: at least 2 samples on sim grid");
    assert(all(diff(aligned.time) > 0), ...
        "Postcondition: time strictly increasing");
    assert(abs(aligned.time(1)) < eps, ...
        "Postcondition: time(1) must be 0");
    assert(size(aligned.butt,1) == N && size(aligned.clubhead,1) == N && ...
           size(aligned.club_quat,1) == N, ...
        "Postcondition: trajectory arrays must match time length");
    assert(all(isfinite(aligned.butt(:))) && all(isfinite(aligned.clubhead(:))), ...
        "Postcondition: positions must be finite");
    assert(all(vecnorm(aligned.butt,2,2) < 5) && ...
           all(vecnorm(aligned.clubhead,2,2) < 5), ...
        "Postcondition: ||r|| must be < 5 m");
    qn = sqrt(sum(aligned.club_quat.^2, 2));
    assert(all(abs(qn - 1) < 1e-6), ...
        "Postcondition: club_quat rows must be unit-norm to 1e-6");
    assert(aligned.impact_idx >= 1 && aligned.impact_idx <= N, ...
        "Postcondition: 1 <= impact_idx <= N");
end


function q_out = local_slerp_resample(t, q_in, t_query)
    % SLERP each row in q_in (Mx4 [w x y z]) for each query time.
    K = numel(t_query);
    q_out = zeros(K, 4);
    for k = 1:K
        tq = t_query(k);
        % Find bracketing indices
        i = find(t <= tq, 1, "last");
        if isempty(i)
            q_out(k, :) = q_in(1, :);
            continue;
        end
        if i >= numel(t)
            q_out(k, :) = q_in(end, :);
            continue;
        end
        j = i + 1;
        u = (tq - t(i)) / (t(j) - t(i));
        q_out(k, :) = local_slerp(q_in(i, :), q_in(j, :), u);
    end
    % Renormalise & sign-canonicalise
    n = sqrt(sum(q_out.^2, 2));
    q_out = q_out ./ n;
    flip = q_out(:, 1) < 0;
    q_out(flip, :) = -q_out(flip, :);
end


function q = local_slerp(q1, q2, u)
    dot_v = sum(q1 .* q2);
    if dot_v < 0
        q2 = -q2;
        dot_v = -dot_v;
    end
    if dot_v > 0.9995
        q = q1 + u * (q2 - q1);
        q = q / norm(q);
        return;
    end
    theta_0 = acos(min(1, max(-1, dot_v)));
    theta   = theta_0 * u;
    sin_t0  = sin(theta_0);
    s1 = sin(theta_0 - theta) / sin_t0;
    s2 = sin(theta) / sin_t0;
    q = s1 * q1 + s2 * q2;
end
