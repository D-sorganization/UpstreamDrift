function opts = default_synth_options()
%DEFAULT_SYNTH_OPTIONS  Canonical options struct for synthesize_target_from_coefficients.
%
%   OPTS = DEFAULT_SYNTH_OPTIONS() returns the default options struct
%   consumed by SYNTHESIZE_TARGET_FROM_COEFFICIENTS.
%
%   Fields:
%     .sample_rate     (1,1) double  Hz, matches simulation. Default 1000.
%     .simulation_time (1,1) double  seconds. Default 0.3.
%     .add_noise       (1,1) logical Add Gaussian position noise. Default false.
%     .noise_sigma_m   (1,1) double  Sigma (metres) when add_noise=true.
%                                    Default 0.001 (1 mm).
%     .subject_id      (1,1) string  Provenance label. Default "synthetic".
%     .trial_id        (1,1) string  Provenance label. Default "synthesizer_v1".
%     .sim_opts        (1,1) struct  Simulation overrides applied on top of
%                                    DEFAULT_SIM_OPTIONS. Default empty struct.
%
%   See also: SYNTHESIZE_TARGET_FROM_COEFFICIENTS, DEFAULT_SIM_OPTIONS.

    opts = struct();
    opts.sample_rate     = 1000;
    opts.simulation_time = 0.3;
    opts.add_noise       = false;
    opts.noise_sigma_m   = 0.001;
    opts.subject_id      = "synthetic";
    opts.trial_id        = "synthesizer_v1";
    opts.sim_opts        = struct();

    required = ["sample_rate","simulation_time","add_noise","noise_sigma_m", ...
                "subject_id","trial_id","sim_opts"];
    assert(all(isfield(opts, required)), ...
        "default_synth_options:missingField", ...
        "Postcondition: default options missing required fields");
end
