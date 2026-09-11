function opts = capture_fit_sim_options(duration_s)
%CAPTURE_FIT_SIM_OPTIONS Independent forward replay over a complete target.
% Keep polynomial actuation enabled across the target horizon. Both sides of
% the model's step gate are one, so a one-second default cannot truncate a
% fitted swing. Source model defaults remain unchanged by per-call overrides.
% Geometry, initial state and coefficient bounds belong to the fit manifest;
% callers add their fixed geometry/initial overrides to input_overrides.
    arguments
        duration_s (1,1) double {mustBeReal, mustBeFinite, mustBePositive}
    end
    opts = default_sim_options();
    opts.simulation_time = duration_s;
    opts.fast_restart = false;
    opts.use_cache = false;
    opts.input_overrides = struct('KillswitchInitialValue', 1, ...
        'KillswitchFinalValue', 1, 'KillswitchStepTime', duration_s);
end
