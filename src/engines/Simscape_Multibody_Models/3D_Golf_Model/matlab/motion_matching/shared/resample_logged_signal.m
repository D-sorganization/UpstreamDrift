function result = resample_logged_signal(signal, target_time, n_columns, solver_time)
%RESAMPLE_LOGGED_SIGNAL Interpolate logged values using their physical clock.
% Missing coverage stays NaN. Numeric array logs require matching solver_time;
% timestamps are never inferred from sample count. No extrapolation or padding.
    arguments
        signal
        target_time (:,1) double {mustBeReal, mustBeFinite}
        n_columns (1,1) double {mustBeInteger, mustBePositive}
        solver_time (:,1) double = []
    end
    assert(~isempty(target_time) && all(diff(target_time) > 0), ...
        'resample_logged_signal:badTargetTime', 'Target clock must increase.');
    [values, source_time] = unpack(signal, solver_time);
    assert(~isempty(source_time), 'resample_logged_signal:missingClock', ...
        'Signal requires recorded timestamps; sample-index interpolation is invalid.');
    source_time = double(source_time(:));
    assert(all(isfinite(source_time)) && all(diff(source_time) > 0), ...
        'resample_logged_signal:badClock', 'Signal clock must be finite and strictly increasing.');
    assert(isnumeric(values) && isreal(values) && ...
        isequal(size(values), [numel(source_time), n_columns]), ...
        'resample_logged_signal:badShape', 'Signal shape must match its clock and requested columns.');
    result = nan(numel(target_time), n_columns);
    if numel(source_time) == 1
        at_sample = target_time == source_time;
        result(at_sample,:) = double(values);
    else
        result = interp1(source_time, double(values), target_time, 'linear', NaN);
    end
end

function [values, time] = unpack(signal, solver_time)
    time = solver_time;
    values = signal;
    if isa(signal, 'timeseries')
        time = signal.Time;
        values = signal.Data;
        if ~signal.IsTimeFirst
            order = [ndims(values), 1:(ndims(values)-1)];
            values = permute(values, order);
        end
        values = reshape(values, numel(time), []);
    elseif isstruct(signal) && isfield(signal, 'Data')
        values = signal.Data;
        if isfield(signal, 'Time'), time = signal.Time; end
    elseif isstruct(signal) && isfield(signal, 'signals') && ...
            isfield(signal.signals, 'values')
        values = signal.signals.values;
        if isfield(signal, 'time'), time = signal.time; end
    end
end
