function result = fit_swing_hybrid(target, options)
%FIT_SWING_HYBRID  Two-stage fit: global gradient-free then SQP polish.
%
%   RESULT = FIT_SWING_HYBRID(TARGET, OPTIONS) is the recommended default
%   entry point for Option 1. Stage 1 spends 70% of
%   options.max_function_evals on options.global_stage ("surrogateopt" by
%   default; "particleswarm" alternative). Stage 2 polishes from the global
%   optimum with fmincon-sqp for the remaining budget. The impact-anchor
%   weight w_a and the regularizer lambda are scheduled per
%   APPROACH.md "Multi-stage schedule".
%
%   See:
%     - INTERFACES.md       (signature contract)
%     - APPROACH.md         (the algorithm)
%     - shared/COST_FUNCTION_SPEC.md
%     - shared/CLUB_IK_SPEC.md
%
%   Preconditions (DbC):
%     - TARGET satisfies validators.mustBeClubTarget
%     - OPTIONS satisfies validators.mustBeOption1Options
%
%   Postconditions:
%     - result.solver == "hybrid"
%     - result.iter_history has a "stage" column with values
%       {"global","polish"}
%     - result.coefficients lies inside (lb, ub)
%     - result has every field in INTERFACES.md "Result struct contract"
%
%   GitHub issue: #026
    arguments
        target  (1,1) struct {validators.mustBeClubTarget}
        options (1,1) struct {validators.mustBeOption1Options} = default_option1_options()
    end
    error("not implemented");
end
