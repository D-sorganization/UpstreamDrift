function tbl = leaderboard(records, varargin)
%LEADERBOARD Rank motion-matching fits using the canonical Metrics schema.
%
%   tbl = leaderboard(records) returns a sorted MATLAB table where each row
%   is a metrics.Metrics record.  Inputs may be:
%     - a string array of paths to JSON files containing Metrics records,
%     - a cell array of metrics.Metrics objects,
%     - a cell/struct array of legacy result structs (auto-converted via
%       metrics.Metrics.fromLegacyStruct for backwards compatibility).
%
%   tbl = leaderboard(records, 'SortBy', name) sorts by the given metric
%   field name (default: 'rmse_clubhead_mm', ascending).
%
%   See METRICS_SCHEMA.md for the canonical field set.

    p = inputParser();
    p.addParameter('SortBy', 'rmse_clubhead_mm', @(x) ischar(x) || isstring(x));
    p.parse(varargin{:});
    sortKey = char(p.Results.SortBy);

    items = local_normalize(records);
    if isempty(items)
        tbl = table();
        return;
    end

    fieldOrder = metrics.Metrics.FIELD_ORDER;
    rows = cell(numel(items), numel(fieldOrder));
    for i = 1:numel(items)
        m = items{i};
        for k = 1:numel(fieldOrder)
            rows{i, k} = m.(fieldOrder(k));
        end
    end
    tbl = cell2table(rows, 'VariableNames', cellstr(fieldOrder));
    if ismember(sortKey, tbl.Properties.VariableNames)
        tbl = sortrows(tbl, sortKey);
    end
end

function items = local_normalize(records)
    if isstring(records)
        items = cell(numel(records), 1);
        for i = 1:numel(records)
            txt = fileread(char(records(i)));
            items{i} = metrics.Metrics.from_json(string(txt));
        end
        return;
    end
    if iscell(records)
        items = cell(numel(records), 1);
        for i = 1:numel(records)
            items{i} = local_coerce(records{i});
        end
        return;
    end
    if isstruct(records)
        items = cell(numel(records), 1);
        for i = 1:numel(records)
            items{i} = local_coerce(records(i));
        end
        return;
    end
    error("leaderboard:badInput", ...
        "records must be string array, cell array, or struct array");
end

function m = local_coerce(item)
    if isa(item, 'metrics.Metrics')
        m = item;
        return;
    end
    if isstruct(item)
        % Heuristic: legacy struct uses short field names.
        if isfield(item, 'rmse_clubhead') || isfield(item, 'rmse_butt')
            m = metrics.Metrics.fromLegacyStruct(item);
        else
            m = metrics.Metrics(item);
        end
        return;
    end
    if isstring(item) || ischar(item)
        m = metrics.Metrics.from_json(string(item));
        return;
    end
    error("leaderboard:badItem", "unsupported record type: %s", class(item));
end
