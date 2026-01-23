function merged = mergeTables(varargin)
    merged = table();
    for i = 1:nargin
        if ~isempty(varargin{i})
            if isempty(merged)
                merged = varargin{i};
            else
                % Outer join on 'time', assuming time is consistent
                merged = outerjoin(merged, varargin{i}, 'Keys', 'time', 'MergeKeys', true);
            end
        end
    end
end
