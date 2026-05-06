function [startTime, stopTime] = frame_horizon(currentState, targetFrame, config)
%FRAME_HORIZON  Compute (StartTime, StopTime) for a single candidate trial.
%
%   [startTime, stopTime] = frame_search.frame_horizon(currentState, ...
%       targetFrame, config) returns the simulation horizon required to
%   advance the model from the current state to one frame past the target.
%
%   Strategy:
%     - currentState.time (default 0) is the time at which the previous
%       frame committed.
%     - targetFrame.time is the time of the next desired frame; if the
%       target struct has no 'time' field, fall back to
%       currentState.time + horizon_frames * median_step from the manifest.
%     - The horizon length is at minimum config.search.horizon_frames *
%       (median step inferred from the manifest) so a single candidate
%       always integrates a non-trivial interval.
%
%   This helper is pure (no Simulink calls) and unit-testable.

    arguments
        currentState struct
        targetFrame struct
        config struct
    end

    if isfield(currentState, 'time') && ~isempty(currentState.time)
        startTime = double(currentState.time);
    else
        startTime = 0.0;
    end

    medianStep = 0.001;  % conservative default
    if isfield(config, 'validation') && ...
            isfield(config.validation, 'median_step_seconds') && ...
            isfinite(config.validation.median_step_seconds) && ...
            config.validation.median_step_seconds > 0
        medianStep = double(config.validation.median_step_seconds);
    end

    horizonFrames = 1;
    if isfield(config, 'search') && isfield(config.search, 'horizon_frames')
        horizonFrames = max(1, double(config.search.horizon_frames));
    end
    minHorizon = horizonFrames * medianStep;

    if isfield(targetFrame, 'time') && ~isempty(targetFrame.time) && ...
            isfinite(targetFrame.time)
        stopTime = double(targetFrame.time);
    else
        stopTime = startTime + minHorizon;
    end

    if stopTime <= startTime
        stopTime = startTime + minHorizon;
    end
end
