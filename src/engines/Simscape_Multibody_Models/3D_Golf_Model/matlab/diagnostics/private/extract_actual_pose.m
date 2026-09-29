function actual = extract_actual_pose(sim_out, joint_names)
%EXTRACT_ACTUAL_POSE  Pull the constraint-resolved pose out of a sim output.
%
%   After the IC solve at t=0, Simscape has projected the requested
%   pose onto the constraint manifold. Recover the resulting joint
%   angles in the same order as JOINT_NAMES, plus key Cartesian
%   markers (butt, clubhead) when the model logs them.

    actual = struct();
    actual.q = nan(numel(joint_names), 1);
    actual.r_butt = [];
    actual.r_clubhead = [];

    % --- 1. Joint angles -----------------------------------------------
    % Preferred path: logsout signals named after the joints.
    logs = [];
    try
        logs = sim_out.logsout;
    catch
        logs = [];
    end

    if ~isempty(logs)
        for i = 1:numel(joint_names)
            name = joint_names{i};
            try
                el = logs.getElement(name);
                ts = el.Values;
                if isa(ts, 'timeseries')
                    actual.q(i) = ts.Data(1);
                elseif isstruct(ts) && isfield(ts, 'Data')
                    actual.q(i) = ts.Data(1);
                end
            catch
                % leave NaN; we'll try fallbacks below
            end
        end
    end

    % Fallback: xFinal operating point.
    if any(isnan(actual.q))
        try
            xf = sim_out.xFinal;
            if isnumeric(xf) && numel(xf) >= numel(joint_names)
                fill = isnan(actual.q);
                actual.q(fill) = xf(find(fill));  %#ok<FNDSB>
            end
        catch
            % no xFinal available
        end
    end

    % --- 2. Cartesian markers ------------------------------------------
    actual.r_butt = local_get_marker(logs, ...
        {'r_butt', 'ButtPosition', 'butt_position', 'butt_xyz'});
    actual.r_clubhead = local_get_marker(logs, ...
        {'r_clubhead', 'ClubheadPosition', 'clubhead_position', 'clubhead_xyz', 'r_ch'});
end

function xyz = local_get_marker(logs, candidates)
    xyz = [];
    if isempty(logs)
        return;
    end
    for k = 1:numel(candidates)
        try
            el = logs.getElement(candidates{k});
            ts = el.Values;
            if isa(ts, 'timeseries')
                d = ts.Data;
            elseif isstruct(ts) && isfield(ts, 'Data')
                d = ts.Data;
            else
                continue;
            end
            if size(d, 1) >= 1 && size(d, 2) == 3
                xyz = d(1, :).';
                return;
            elseif numel(d) >= 3
                xyz = d(1:3);
                xyz = xyz(:);
                return;
            end
        catch
            % keep searching
        end
    end
end
