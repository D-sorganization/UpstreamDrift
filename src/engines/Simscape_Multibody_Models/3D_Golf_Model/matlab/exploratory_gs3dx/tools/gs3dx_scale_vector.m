function scaled = gs3dx_scale_vector(vec, factor, axis_idx)
%GS3DX_SCALE_VECTOR  Pure mathematical scaling of a 3D vector along a single axis.
%
%   SCALED = GS3DX_SCALE_VECTOR(VEC, FACTOR, AXIS_IDX) scales the element of
%   the 1x3 real finite numeric vector VEC at index AXIS_IDX (default 3) by the
%   positive finite real scalar FACTOR, preserving the other two elements.
%
%   Inputs:
%     VEC      - 1x3 (or 3x1) real finite numeric vector [x, y, z]
%     FACTOR   - positive finite real scalar scale factor
%     AXIS_IDX - integer axis index in {1, 2, 3}. Default: 3 (longitudinal z)
%
%   Output:
%     SCALED   - 1x3 double row vector with element AXIS_IDX scaled by FACTOR

    arguments
        vec
        factor
        axis_idx = 3
    end

    % 1. Validate vector
    assert(isnumeric(vec) && isreal(vec) && isvector(vec) && all(isfinite(vec(:))) && numel(vec) == 3, ...
        'gs3dx:scale_vector:InvalidVector', ...
        'Input vector must be a 1x3 real finite numeric vector');

    % 2. Validate scale factor
    assert(isnumeric(factor) && isreal(factor) && isscalar(factor) && isfinite(factor) && factor > 0, ...
        'gs3dx:scale_vector:InvalidScaleFactor', ...
        'Scale factor must be a positive finite real scalar');

    % 3. Validate axis index
    assert(isnumeric(axis_idx) && isreal(axis_idx) && isscalar(axis_idx) && isfinite(axis_idx) && ...
        ismember(axis_idx, [1, 2, 3]) && axis_idx == fix(axis_idx), ...
        'gs3dx:scale_vector:InvalidAxis', ...
        'Axis index must be an integer in {1, 2, 3}');

    % 4. Pure linear scaling
    scaled = double(vec(:).');
    scaled(axis_idx) = scaled(axis_idx) * factor;
end
