function result = fit_swing_hybrid(target, options)
%FIT_SWING_HYBRID  Two-stage fit: NN-surrogate warm start, then fmincon polish.
%
%   RESULT = FIT_SWING_HYBRID(TARGET, OPTIONS) is the canonical Option 2 ->
%   Option 1 handoff per APPROACH.md "Hybrid: handoff to Option 1". Stage 1
%   calls Python's `fit_swing_via_surrogate` (issue #029) via `pyrun` to get
%   a warm-start coefficient vector. Stage 2 then runs `fit_swing_fmincon`
%   (issue #024) seeded with that warm start. The full result struct from
%   the polish stage is returned, with both phases recorded under
%   `result.surrogate_phase` and `result.fmincon_phase` for provenance.
%
%   If `options.skip_polish_tol_m` is set and the surrogate's reported loss
%   is already below that tolerance, the polish stage is skipped and the
%   surrogate's warm-start coefficients are returned as the final answer.
%
%   Preconditions (DbC):
%     - TARGET satisfies validators.mustBeClubTarget
%     - OPTIONS satisfies validators.mustBeOption1Options
%     - OPTIONS.surrogate_checkpoint, when present, is a path that exists
%
%   Postconditions:
%     - result.solver == "surrogate+fmincon"
%     - result.surrogate_phase is a struct with fields {coefficients,
%       final_loss, duration_s}
%     - result.fmincon_phase is a struct (the full Option 1 result) or []
%       if the polish was skipped
%     - result.coefficients lies inside (lb, ub)
%
%   GitHub issue: #4000 / #031.
    arguments
        target  (1,1) struct {validators.mustBeClubTarget}
        options (1,1) struct {validators.mustBeOption1Options} = default_option1_options()
    end

    t_start = tic;

    % ---- 1. Surrogate warm start (Python via pyrun) ------------------------
    surrogate_phase = call_python_surrogate_invert(target, options);

    theta_warm = double(surrogate_phase.coefficients(:));

    % ---- 2. Decide whether to skip polish ----------------------------------
    skip_tol = local_get(options, "skip_polish_tol_m", -inf);
    surr_loss = double(local_get(surrogate_phase, "final_loss", inf));
    skip_polish = isfinite(skip_tol) && surr_loss <= skip_tol;

    if skip_polish
        fmincon_phase = [];
        coefficients = theta_warm;
        final_rmse_m = sqrt(max(0, surr_loss));
        iter_history = table('Size', [0, 4], ...
            'VariableTypes', {'double','double','double','double'}, ...
            'VariableNames', {'iteration','fval','firstorderopt','stepsize'});
        exitflag = 99;          % sentinel: skipped polish
        output_struct = struct('message', "polish skipped: warm start below skip_polish_tol_m");
    else
        % ---- 3. Polish via fit_swing_fmincon -------------------------------
        polish_opts = options;
        polish_opts.initial_theta = theta_warm;
        fmincon_phase = fit_swing_fmincon(target, polish_opts);

        coefficients = fmincon_phase.coefficients(:);
        final_rmse_m = double(fmincon_phase.final_rmse_m);
        iter_history = fmincon_phase.iter_history;
        exitflag = fmincon_phase.exitflag;
        output_struct = fmincon_phase.output;
    end

    % ---- 4. Assemble combined result struct --------------------------------
    result = struct();
    result.coefficients     = coefficients;
    result.final_rmse_m     = final_rmse_m;
    result.solver           = "surrogate+fmincon";
    result.surrogate_phase  = surrogate_phase;
    result.fmincon_phase    = fmincon_phase;
    result.iter_history     = iter_history;
    result.exitflag         = exitflag;
    result.output           = output_struct;
    result.duration_s       = toc(t_start);
    result.target_hash      = local_safe_target_hash(target);
    result.matlab_version   = string(version);
    result.timestamp_utc    = string(datetime("now","TimeZone","UTC", ...
                                "Format","yyyy-MM-dd'T'HH:mm:ss'Z'"));
    result.options          = options;
    result.cache_hit        = false;

    % ---- 5. Postconditions -------------------------------------------------
    assert(result.solver == "surrogate+fmincon", ...
        "fit_swing_hybrid:postSolver", ...
        "Postcondition: result.solver must be ""surrogate+fmincon""");
    assert(isstruct(result.surrogate_phase), ...
        "fit_swing_hybrid:postSurrogatePhase", ...
        "Postcondition: result.surrogate_phase must be a struct");
    assert(isfinite(result.final_rmse_m) && result.final_rmse_m >= 0, ...
        "fit_swing_hybrid:postRmse", ...
        "Postcondition: final_rmse_m must be finite and non-negative");
end

% =====================================================================
function v = local_get(s, name, default)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = default;
    end
end

% =====================================================================
function h = local_safe_target_hash(target)
    h = "unknown";
    try
        t = target;
        if isfield(t, "source"), t = rmfield(t, "source"); end
        json = jsonencode(t);
        bytes = uint8(char(json));
        md = java.security.MessageDigest.getInstance('SHA-256');
        md.update(bytes);
        digest = typecast(md.digest(), 'uint8');
        h = string(lower(reshape(dec2hex(digest, 2).', 1, [])));
    catch
        h = "unknown";
    end
end
