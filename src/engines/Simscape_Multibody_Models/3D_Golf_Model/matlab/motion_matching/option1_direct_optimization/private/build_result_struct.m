function result = build_result_struct(args)
%BUILD_RESULT_STRUCT  Assemble the canonical Option 1 result struct.
%
%   RESULT = BUILD_RESULT_STRUCT(ARGS) returns the provenance-bearing
%   result struct described in shared/CODING_STANDARDS.md
%   "Provenance and reproducibility" and INTERFACES.md "Result struct
%   contract".
%
%   ARGS is a 1x1 struct with the following fields (all required):
%       coefficients     (d,1) double
%       final_cost_terms struct from compute_cost
%       final_sim_out    struct from simulate_with_coefficients
%       solver           string
%       solver_options   struct (the optimoptions used)
%       options          struct (full Option 1 options)
%       target           struct (input target)
%       exitflag         integer
%       output           struct (raw fmincon output)
%       iter_history     table
%       duration_s       scalar double
%       start_points     d x N double or []
%       start_costs      1 x N double or []
%       cache_hit        logical
%
%   Postconditions:
%     - Every field listed in INTERFACES.md "Result struct contract" is
%       present on the returned struct.
%     - result.final_rmse_m is finite and >= 0.
%
%   GitHub issue: #024 / #3993.
    arguments
        args (1,1) struct
    end

    required = ["coefficients","final_cost_terms","final_sim_out","solver", ...
                "solver_options","options","target","exitflag","output", ...
                "iter_history","duration_s","start_points","start_costs", ...
                "cache_hit"];
    missing = setdiff(required, string(fieldnames(args)));
    if ~isempty(missing)
        error("build_result_struct:missingArg", ...
              "Missing args fields: %s", strjoin(missing, ", "));
    end

    result = struct();
    result.coefficients       = args.coefficients(:);
    result.final_rmse_m       = local_rmse_from_terms(args.final_cost_terms);
    result.final_total_work_J = local_safe_total_work(args.final_sim_out);
    result.final_cost_terms   = args.final_cost_terms;
    result.solver             = string(args.solver);
    result.solver_options     = args.solver_options;
    result.target_hash        = local_hash_target(args.target);
    result.git_commit         = local_safe_git_commit();
    result.matlab_version     = string(version);
    result.duration_s         = double(args.duration_s);
    result.timestamp_utc      = string(datetime("now","TimeZone","UTC", ...
                                "Format","yyyy-MM-dd'T'HH:mm:ss'Z'"));
    result.iter_history       = args.iter_history;
    result.exitflag           = args.exitflag;
    result.output             = args.output;
    result.start_points       = args.start_points;
    result.start_costs        = args.start_costs;
    result.cache_hit          = logical(args.cache_hit);
    result.options            = args.options;

    assert(isfinite(result.final_rmse_m) && result.final_rmse_m >= 0, ...
        "build_result_struct:badRmse", ...
        "Postcondition: final_rmse_m must be finite and non-negative");
end

function rmse = local_rmse_from_terms(terms)
    % terms.position is the weighted mean per-frame squared distance
    % (sum of butt^2 + ch^2). The reported RMSE is the unweighted
    % per-frame RMS in metres of that squared sum. Since we only have the
    % weighted value, undo the weight if w_position was applied; the
    % caller passes the raw terms.position from compute_cost.
    if ~isstruct(terms) || ~isfield(terms, "position")
        rmse = NaN;
        return;
    end
    val = double(terms.position);
    if ~isfinite(val) || val < 0
        rmse = NaN;
    else
        rmse = sqrt(val);
    end
end

function w = local_safe_total_work(sim_out)
    w = NaN;
    try
        w = compute_total_work(sim_out);
    catch
        w = NaN;
    end
end

function h = local_hash_target(target)
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
