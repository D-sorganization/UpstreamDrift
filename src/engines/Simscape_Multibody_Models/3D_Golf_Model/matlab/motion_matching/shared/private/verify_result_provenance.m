function [ok, missing] = verify_result_provenance(result)
%VERIFY_RESULT_PROVENANCE  Check a result struct has provenance fields.
%
%   [OK, MISSING] = VERIFY_RESULT_PROVENANCE(RESULT) returns OK=true when
%   RESULT contains all the fields required by CODING_STANDARDS.md
%   §"Provenance and reproducibility":
%
%       coefficients, final_rmse_m, final_total_work_J, solver,
%       solver_options, target_hash, git_commit, matlab_version,
%       duration_s, timestamp_utc
%
%   When OK=false, MISSING is a string row vector listing the absent
%   fields so the caller can warn with a precise message. This is used by
%   leaderboard.m to skip-with-warning malformed result structs rather
%   than abort the whole scan.
%
%   Preconditions:
%     - RESULT is a 1x1 struct.
%   Postconditions:
%     - OK is logical scalar.
%     - MISSING is a 1xN string (possibly 1x0); empty iff OK is true.
    arguments
        result (1,1) struct
    end

    required = ["coefficients", "final_rmse_m", "final_total_work_J", ...
                "solver", "solver_options", "target_hash", "git_commit", ...
                "matlab_version", "duration_s", "timestamp_utc"];
    actual  = string(fieldnames(result));
    missing = setdiff(required, actual, 'stable');
    ok      = isempty(missing);

    assert(islogical(ok) && isscalar(ok), ...
        "verify_result_provenance:postcondition", "ok must be logical scalar");
    assert(isstring(missing), ...
        "verify_result_provenance:postcondition", "missing must be string");
end
