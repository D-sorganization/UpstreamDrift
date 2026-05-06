function options = default_option1_options()
%DEFAULT_OPTION1_OPTIONS  Canonical options struct for Option 1 fits.
%
%   OPTIONS = DEFAULT_OPTION1_OPTIONS() returns the default options struct
%   consumed by every fit_swing_* entry point. Override individual fields
%   and pass the result to fit_swing_*.
%
%   See INTERFACES.md "default_option1_options" for the full table of
%   fields, types, and defaults.
%
%   Postconditions:
%     - All fields documented in INTERFACES.md are present.
%     - opts is a 1x1 struct.
%
%   GitHub issue: #024 / #3993.

    options = struct();
    options.solver                 = "fmincon";
    options.schedule               = "flat";
    options.cost                   = default_cost_options();
    options.sim                    = default_sim_options();
    options.cold_start_strategy    = "zeros";
    options.initial_theta          = [];        % numeric vector, [] = use cold-start
    options.multistart_n           = uint32(8);
    options.max_iter               = uint32(200);
    options.max_iterations         = uint32(200);    % alias used by INTERFACES.md
    options.max_function_evals     = uint32(5000);
    options.fd_central             = false;
    options.tol_fun                = 1e-6;
    options.tol_x                  = 1e-8;
    options.use_parallel           = false;
    options.num_workers            = uint32(1);
    options.rng_seed               = uint32(42);
    options.use_cache              = true;
    options.cache_dir              = "results/cache";
    options.verbosity              = "Normal";
    options.dashboard              = false;
    options.dashboard_refresh_hz   = 5;
    options.penalty_on_sim_failure = 1e9;
    options.max_sim_failures       = uint32(50);
    options.global_stage           = "surrogateopt";
    options.algorithm              = "sqp";
    options.display                = "iter";
    options.results_dir            = "";
    options.output_fcn             = [];        % optional fmincon OutputFcn

    required = ["solver", "schedule", "cost", "sim", "cold_start_strategy", ...
                "initial_theta", "max_iter", "max_function_evals", ...
                "tol_fun", "tol_x", "rng_seed", "verbosity", ...
                "penalty_on_sim_failure", "max_sim_failures", ...
                "algorithm", "display", "output_fcn"];
    assert(all(isfield(options, required)), ...
        "default_option1_options:missingField", ...
        "Postcondition: default options missing required fields");
end
