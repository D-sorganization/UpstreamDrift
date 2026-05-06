function opts = default_sim_options()
%DEFAULT_SIM_OPTIONS  Canonical options struct for simulate_with_coefficients.
%
%   OPTS = DEFAULT_SIM_OPTIONS() returns the default options struct consumed
%   by SIMULATE_WITH_COEFFICIENTS. Override individual fields and pass the
%   result to the wrapper.
%
%   Fields:
%     .model_name        char/string  Simulink model name. Default
%                                     "GolfSwing3D_Kinetic".
%     .simulation_time   (1,1) double Stop time in seconds. Default 0.3.
%     .sample_rate       (1,1) double Output sample rate in Hz. Default 1000.
%     .solver            string       Simulink solver. Default "ode23t".
%     .fast_restart      (1,1) logical Use FastRestart. Default true.
%     .parallel_safe     (1,1) logical Disable FastRestart for parsim/parfor.
%                                     Default false.
%     .verbosity         string       "Silent" | "Normal" | "Verbose" | "Debug".
%                                     Default "Normal".
%     .cache_dir         string       Optional cache directory; "" disables.
%                                     Default "".
%     .use_cache         (1,1) logical Master switch. Default false.
%     .stop_on_error     (1,1) logical Re-throw simulation errors when true,
%                                     else mark sim_out.solver_status="failed".
%                                     Default true.
%     .joint_names       string array Optional override for joint ordering;
%                                     when empty (default) the joints are
%                                     discovered via getPolynomialParameterInfo.
%
%   Postconditions:
%     - All documented fields are present.
%     - opts is a 1x1 struct.
%
%   See also: SIMULATE_WITH_COEFFICIENTS.

    opts = struct();
    opts.model_name        = "GolfSwing3D_Kinetic";
    opts.simulation_time   = 0.3;
    opts.sample_rate       = 1000;
    opts.solver            = "ode23t";
    opts.fast_restart      = true;
    opts.parallel_safe     = false;
    opts.verbosity         = "Normal";
    opts.cache_dir         = "";
    opts.use_cache         = false;
    opts.stop_on_error     = true;
    opts.joint_names       = string.empty(1, 0);
    % Per-call overrides on model-workspace variables — used by Stage-1
    % starting-pose tooling to perturb the initial pose without editing
    % the source MAT.  Each field is a model-workspace variable name and
    % the value is the override to apply via setVariable.
    opts.input_overrides   = struct();

    % Postconditions
    required = ["model_name", "simulation_time", "sample_rate", "solver", ...
                "fast_restart", "parallel_safe", "verbosity", "cache_dir", ...
                "use_cache", "stop_on_error", "joint_names", "input_overrides"];
    assert(all(isfield(opts, required)), ...
        "default_sim_options:missingField", ...
        "Postcondition: default options missing required fields");
end
