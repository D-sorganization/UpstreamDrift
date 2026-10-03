function w = gs3dx_ik_gap_weights(names, base_gap_weight, overrides)
%GS3DX_IK_GAP_WEIGHTS Compute per-target gap residual scale factors (#10979).
%
%   W = GS3DX_IK_GAP_WEIGHTS(NAMES, BASE_GAP_WEIGHT, OVERRIDES) returns a 1 x NT
%   double vector of absolute gap residual scale factors for the NT position
%   target names in NAMES.
%
%   Inputs:
%     NAMES            Cell array of character vectors or string array of unique,
%                      non-empty target identifiers. Rejects missing or empty strings.
%     BASE_GAP_WEIGHT  Finite real numeric scalar in [0, 1] specifying the baseline
%                      residual scale for gap-filled position targets.
%     OVERRIDES        Empty ([] or empty struct()) or a scalar struct whose field
%                      names belong to NAMES and whose values are finite real numeric
%                      scalars in [0, 1]. Rejects empty cells or strings. Targets
%                      not specified in OVERRIDES inherit BASE_GAP_WEIGHT.
%
%   Output:
%     W                1 x NT double row vector of per-target gap residual scales.

    % Validate names: unique, non-empty, handle string arrays and missing strings
    if isstring(names)
        if any(ismissing(names), "all")
            error('gs3dx:ik', 'names contains missing string elements');
        end
        names = cellstr(names);
    end

    if ~iscellstr(names) || isempty(names) || ~isvector(names)
        error('gs3dx:ik', 'names must be a non-empty vector of strings or cell array of char vectors');
    end

    for k = 1:numel(names)
        str_k = names{k};
        if ~isrow(str_k) || ~isvarname(str_k)
            error('gs3dx:ik', 'names elements must be non-empty identifiers');
        end
    end

    if numel(names) ~= numel(unique(names))
        error('gs3dx:ik', 'names elements must be unique');
    end

    % Validate base_gap_weight: numeric finite real scalar in [0, 1]
    if ~isnumeric(base_gap_weight) || ~isscalar(base_gap_weight) || ...
            ~isreal(base_gap_weight) || ~isfinite(base_gap_weight) || ...
            base_gap_weight < 0 || base_gap_weight > 1
        error('gs3dx:ik', 'base_gap_weight must be a finite real numeric scalar in [0, 1]');
    end
    base_gap_weight = double(base_gap_weight);

    nt = numel(names);
    w = repmat(base_gap_weight, 1, nt);

    if nargin < 3
        return;
    end

    % Validate overrides: only [] or empty struct or scalar struct; reject empty cells/strings
    if iscell(overrides)
        error('gs3dx:ik', 'overrides must not be a cell array');
    end
    if ischar(overrides) || isstring(overrides)
        error('gs3dx:ik', 'overrides must not be a string or character array');
    end

    if isempty(overrides)
        if (isnumeric(overrides) && isequal(size(overrides), [0, 0])) || ...
           (isstruct(overrides) && isempty(fieldnames(overrides)))
            return;
        else
            error('gs3dx:ik', 'overrides must be [], empty struct, or a scalar struct');
        end
    end

    if ~isstruct(overrides) || ~isscalar(overrides)
        error('gs3dx:ik', 'overrides must be empty or a scalar struct');
    end

    fnames = fieldnames(overrides);
    for i = 1:numel(fnames)
        fn = fnames{i};
        if ~ismember(fn, names)
            error('gs3dx:ik', 'unknown gap_target_weight field %s', fn);
        end
        val = overrides.(fn);
        if ~isnumeric(val) || ~isscalar(val) || ~isreal(val) || ~isfinite(val) || val < 0 || val > 1
            error('gs3dx:ik', 'gap_target_weight for %s must be a finite real numeric scalar in [0, 1]', fn);
        end
    end

    for k = 1:nt
        nm = names{k};
        if isfield(overrides, nm)
            w(k) = double(overrides.(nm));
        end
    end
end
