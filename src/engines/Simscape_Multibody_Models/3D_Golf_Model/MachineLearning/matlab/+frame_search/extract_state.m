function nextState = extract_state(simOut, previousState, config)
%EXTRACT_STATE  Build the carry-forward state struct after a candidate run.
%
%   nextState = frame_search.extract_state(simOut, previousState, config)
%   harvests the (q, qd) joint configuration plus any final-state struct
%   from a Simulink.SimulationOutput so the next frame can resume from the
%   exact end of this one.
%
%   Output struct fields:
%     .xFinal        Simulink final-state struct (if SaveFinalState was on).
%     .q             column vector of joint positions (one per manifest joint)
%                    when those signals were logged, else empty.
%     .qd            column vector of joint velocities, else empty.
%     .time          last-sample time of simOut (s).
%     .frame_index   carried/incremented from previousState.
%     .starting_state_file
%                    preserved from previousState so the manifest path is
%                    available on subsequent frames if xFinal is empty.

    arguments
        simOut
        previousState struct
        config struct
    end

    nextState = previousState;

    % Carry/extract xFinal if we have it.
    if isprop(simOut, 'xFinal') || isfield(simOut, 'xFinal')
        try
            nextState.xFinal = simOut.xFinal;
        catch
        end
    end

    % Last-sample time for horizon planning on the next frame.
    nextState.time = local_last_time(simOut);

    % Optional q / qd snapshots from joint position/velocity columns when
    % those columns are listed in the manifest's input section.
    [qNames, qdNames] = local_state_column_names(config);
    nextState.q = local_pull_state_vector(simOut, qNames);
    nextState.qd = local_pull_state_vector(simOut, qdNames);

    if isfield(nextState, 'frame_index')
        % evaluate_candidate_step already incremented; do not double-count.
    else
        nextState.frame_index = 1;
    end
end

function t = local_last_time(simOut)
    t = NaN;
    candidates = {'tout', 'time'};
    for k = 1:numel(candidates)
        try
            if isprop(simOut, candidates{k}) || isfield(simOut, candidates{k})
                v = simOut.(candidates{k});
                if isnumeric(v) && ~isempty(v)
                    t = double(v(end));
                    return;
                end
            end
        catch
        end
    end
    try
        if isprop(simOut, 'logsout') && ~isempty(simOut.logsout)
            ls = simOut.logsout;
            if numElements(ls) >= 1
                el = ls{1};
                if isprop(el, 'Values') && isprop(el.Values, 'Time')
                    times = el.Values.Time;
                    if ~isempty(times)
                        t = double(times(end));
                    end
                end
            end
        end
    catch
    end
end

function [qNames, qdNames] = local_state_column_names(config)
    qNames = {};
    qdNames = {};
    if ~isfield(config, 'columns')
        return;
    end
    cols = config.columns;
    if isfield(cols, 'joint_position_columns')
        qNames = local_to_cellstr(cols.joint_position_columns);
    end
    if isfield(cols, 'joint_velocity_columns')
        qdNames = local_to_cellstr(cols.joint_velocity_columns);
    end
end

function out = local_to_cellstr(v)
    if iscell(v)
        out = cellfun(@char, v, 'UniformOutput', false);
    elseif isstring(v)
        out = cellstr(v);
    elseif ischar(v)
        out = {v};
    else
        out = {};
    end
end

function vec = local_pull_state_vector(simOut, names)
    if isempty(names)
        vec = [];
        return;
    end
    vec = nan(numel(names), 1);
    for idx = 1:numel(names)
        try
            info = frame_search.parse_target_column(string(names{idx}));
            vec(idx) = frame_search.lookup_signal_value(simOut, info, []);
        catch
            vec(idx) = NaN;
        end
    end
end
