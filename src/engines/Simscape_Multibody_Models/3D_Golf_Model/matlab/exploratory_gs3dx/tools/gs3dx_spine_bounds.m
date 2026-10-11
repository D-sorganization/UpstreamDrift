function b = gs3dx_spine_bounds(jp, layout, reference, bend_deg, twist_deg)
%GS3DX_SPINE_BOUNDS  Bilateral/axial spine excursion bounds in native coordinates.
%   B = GS3DX_SPINE_BOUNDS(JP, LAYOUT, REFERENCE, BEND_DEG, TWIST_DEG)
%   computes independent coordinate bounds for Spine Tilt (Rx.q, Ry.q) and Torso
%   Kinetically Driven (Rz.q) around REFERENCE.
%
%   When both BEND_DEG == 0 and TWIST_DEG == 0, returns ACTIVE=false and empty bounds.
%   Otherwise validates scalar real finite nonnegative limits, resolves coordinate
%   indices from JP and LAYOUT prefix sums, and bounds resolved coordinates by
%   REFERENCE +/- DEG2RAD(LIMIT), leaving +/-Inf elsewhere. A zero limit for one
%   group means no bounds (i.e. +/-Inf) for that group.
%
%   Errors with ID 'gs3dx:ik:spine' for duplicate/missing/malformed units/layout,
%   nonfinite reference, dimension mismatch, or invalid inputs.

    % Validate limits
    assert(isnumeric(bend_deg) && isscalar(bend_deg) && isreal(bend_deg) && ...
        isfinite(bend_deg) && bend_deg >= 0, ...
        'gs3dx:ik:spine', 'bend_deg must be a scalar real finite nonnegative number');
    assert(isnumeric(twist_deg) && isscalar(twist_deg) && isreal(twist_deg) && ...
        isfinite(twist_deg) && twist_deg >= 0, ...
        'gs3dx:ik:spine', 'twist_deg must be a scalar real finite nonnegative number');

    bend_deg = double(bend_deg);
    twist_deg = double(twist_deg);

    if bend_deg == 0 && twist_deg == 0
        b = struct('active', false, ...
                   'indices', zeros(0, 1), ...
                   'keys', string.empty(0, 1), ...
                   'reference', zeros(0, 1), ...
                   'lower', zeros(0, 1), ...
                   'upper', zeros(0, 1));
        return;
    end

    % Validate table and layout
    assert(istable(jp) && all(ismember({'ID', 'BlockPath', 'Unit'}, jp.Properties.VariableNames)), ...
        'gs3dx:ik:spine', 'Joint table must include ID, BlockPath, and Unit');
    assert(isstruct(layout) && isfield(layout, 'key') && isfield(layout, 'n'), ...
        'gs3dx:ik:spine', 'Layout must be a struct with key and n fields');

    % Prefix sums and total dimension
    n_layout = [layout.n];
    assert(isnumeric(n_layout) && all(isfinite(n_layout)) && all(n_layout >= 1) && all(n_layout==fix(n_layout)), ...
        'gs3dx:ik:spine', 'Layout n must contain positive finite integers');
    n = sum(n_layout);

    % Validate reference
    assert(isnumeric(reference) && isvector(reference) && isreal(reference) && all(isfinite(reference(:))), ...
        'gs3dx:ik:spine', 'Reference must be numeric, real, and finite');
    assert(numel(reference) == n, ...
        'gs3dx:ik:spine', 'Reference length must equal sum(layout.n)');
    reference = double(reference(:));

    b = struct('active', true, ...
               'indices', zeros(3, 1), ...
               'keys', strings(3, 1), ...
               'reference', reference, ...
               'lower', -inf(n, 1), ...
               'upper', inf(n, 1));

    starts = [0, cumsum(n_layout)];
    keys = string({layout.key});

    % Specs: [Path pattern, ID suffix, Limit]
    specs = {
        "Spine Tilt", ".Rx.q", bend_deg;
        "Spine Tilt", ".Ry.q", bend_deg;
        "Torso Kinetically Driven", ".Rz.q", twist_deg
    };

    block_paths = string(jp.BlockPath);
    ids = string(jp.ID);
    units = string(jp.Unit);

    for s = 1:size(specs, 1)
        path_str = specs{s, 1};
        id_suffix = specs{s, 2};
        limit_val = specs{s, 3};

        m = contains(block_paths, path_str) & endsWith(ids, id_suffix);
        assert(nnz(m) == 1, 'gs3dx:ik:spine', ...
            'Expected exactly one match for %s %s', path_str, id_suffix);
        assert(units(m) == "deg", 'gs3dx:ik:spine', ...
            'Native joint unit must be deg for %s %s', path_str, id_suffix);

        matched_id = ids(m);
        coord_key = extractBefore(matched_id, '.q');
        k = find(keys == coord_key);
        assert(numel(k) == 1 && layout(k).n == 1, 'gs3dx:ik:spine', ...
            'Target coordinate %s must resolve to a single scalar independent variable', coord_key);

        idx = starts(k) + 1;
        b.indices(s) = idx;
        b.keys(s) = coord_key;

        if limit_val > 0
            delta = deg2rad(limit_val);
            b.lower(idx) = reference(idx) - delta;
            b.upper(idx) = reference(idx) + delta;
        end
    end
end
