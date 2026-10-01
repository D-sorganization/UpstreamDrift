function flat = gs3dx_flatten_bus(bus, time_grid)
%GS3DX_FLATTEN_BUS  Flatten a logged bus of timeseries onto one uniform time grid.
%
%   FLAT = GS3DX_FLATTEN_BUS(BUS, TIME_GRID) walks a (nested) struct whose
%   leaves are timeseries objects and returns a struct with fields:
%     .time    (N,1) the TIME_GRID column
%     .names   (1,M) string of leaf paths joined with "/"
%     .data    {1,M} cell; each entry is (N, k) double, linearly interpolated
%              onto TIME_GRID (k = number of flattened channels of that leaf)
%
%   A uniform grid makes runs with different variable-step solvers directly
%   comparable (GS3DX_COMPARE) and keeps stored baselines small.
%
%   Preconditions: TIME_GRID is a finite, strictly increasing vector.
%   Postconditions: numel(flat.names) == numel(flat.data); every data entry
%   has numel(TIME_GRID) rows.

    arguments
        bus
        time_grid (:,1) double {mustBeFinite}
    end
    assert(all(diff(time_grid) > 0), 'gs3dx:flatten', ...
        'Precondition: time_grid must be strictly increasing');

    flat = struct('time', time_grid, 'names', strings(1, 0), 'data', {cell(1, 0)});
    flat = local_walk(bus, "", time_grid, flat);

    assert(numel(flat.names) == numel(flat.data), 'gs3dx:flatten', ...
        'Postcondition: names/data length mismatch');
end

function flat = local_walk(node, prefix, grid, flat)
    if isa(node, 'timeseries')
        t = double(node.Time(:));
        d = double(squeeze(node.Data));
        if isscalar(t)
            return;                         % constant-sample leaves carry no trajectory
        end
        if size(d, 1) ~= numel(t)           % (k, N) or (a, b, N) layouts
            d = reshape(d, [], numel(t)).';
        end
        [t, iu] = unique(t, 'last');        % variable-step logs repeat zero-crossing times
        d = d(iu, :);
        flat.names(end+1) = prefix;
        flat.data{end+1}  = interp1(t, d, grid, 'linear', 'extrap');
    elseif isstruct(node)
        f = fieldnames(node);
        for k = 1:numel(f)
            child = f{k};
            if prefix == ""
                p = string(child);
            else
                p = prefix + "/" + child;
            end
            flat = local_walk(node.(child), p, grid, flat);
        end
    end
end
