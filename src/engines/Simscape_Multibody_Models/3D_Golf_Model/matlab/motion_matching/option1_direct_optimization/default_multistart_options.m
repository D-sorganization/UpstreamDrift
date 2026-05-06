function options = default_multistart_options()
%DEFAULT_MULTISTART_OPTIONS  Canonical options struct for fit_swing_multistart.
%
%   OPTIONS = DEFAULT_MULTISTART_OPTIONS() returns the default options for
%   the parallel multi-start driver. The struct extends the canonical
%   Option 1 options (see DEFAULT_OPTION1_OPTIONS) with the multistart-
%   specific fields documented in INTERFACES.md.
%
%   Postconditions:
%     - All fields documented in the issue #3994 are present.
%     - opts is a 1x1 struct.
%     - opts.fmincon_options satisfies the Option 1 options contract.
%
%   GitHub issue: #025 / #3994.

    options                       = default_option1_options();
    options.solver                = "multistart";
    options.n_starts              = uint32(8);
    options.starting_strategy     = "sobol";    % "random" | "latin_hypercube" | "sobol"
    options.parallel_pool_size    = "auto";      % "auto" | numeric
    options.parallel_method       = "parfor";   % "parsim" | "parfor"
    options.parallel              = true;        % alias used by issue body
    options.seed                  = uint32(42);
    options.fmincon_options       = default_option1_options();

    required = ["n_starts","starting_strategy","parallel_pool_size", ...
                "parallel_method","seed","fmincon_options","results_dir"];
    assert(all(isfield(options, required)), ...
        "default_multistart_options:missingField", ...
        "Postcondition: default options missing required fields");
end
