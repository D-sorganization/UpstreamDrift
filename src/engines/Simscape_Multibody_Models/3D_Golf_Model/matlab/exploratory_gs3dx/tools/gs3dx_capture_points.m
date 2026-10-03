function [points_si, missing_mask] = gs3dx_capture_points(points, units, residuals, observed_mask)
%GS3DX_CAPTURE_POINTS  Convert C3D homogeneous points to SI Z-up with residual masking (#10985, #11011).
%
%   [POINTS_SI, MISSING_MASK] = GS3DX_CAPTURE_POINTS(POINTS, UNITS, RESIDUALS)
%   [POINTS_SI, MISSING_MASK] = GS3DX_CAPTURE_POINTS(POINTS, UNITS, RESIDUALS, OBSERVED_MASK)
%   converts homogeneous 4 x N x T points from ezc3d to 3 x N x T SI metres in
%   Simscape Z-up coordinates (x, -z, y), masking invalid samples to NaN based on
%   coordinate finiteness, ezc3d residuals, and optional decoded source observation mask.
%
%   Design-by-Contract rules:
%     - POINTS must be a real numeric 4 x N x T array (XYZ1 homogeneous).
%       Row 4 is homogeneous 1 and is never treated as a residual.
%     - UNITS must be a scalar string or char vector from the supported set:
%         'm'  : factor 1.0
%         'mm' : factor 1e-3 (0.001)
%         'cm' : factor 1e-2 (0.01)
%       Any other unit string is rejected fail-closed.
%     - RESIDUALS must be a real numeric 1 x N x T array matching POINTS in
%       marker count N and frame count T.
%     - OBSERVED_MASK (optional) is a logical array representing original decoded
%       source validity:
%         * Defaults to empty (no observation mask applied).
%         * Expected shape is 1 x N x T (singleton handling supported: N x 1 for
%           single frame, 1 x T or T x 1 for single marker).
%         * Non-logical types, arrays with > 3 dimensions, and dimension mismatches
%           are rejected fail-closed before numerical operations.
%         * False entries augment MISSING_MASK and force all 3 XYZ coordinates to NaN.
%         * True entries NEVER revive an otherwise invalid sample (negative residual,
%           nonfinite residual, or nonfinite coordinate).
%     - A marker sample (n, t) is deemed missing/invalid if:
%         1. residuals(1, n, t) < 0 (ezc3d invalid residual convention)
%         2. residuals(1, n, t) is nonfinite (NaN or Inf)
%         3. any coordinate in points(1:3, n, t) is nonfinite (NaN or Inf)
%         4. observed_mask(1, n, t) is false (when observed_mask is supplied)
%     - Invalid samples have all 3 coordinates in POINTS_SI set to NaN.
%       Missing values are NEVER replaced by zeros.
%     - MISSING_MASK is a 1 x N x T logical array, true for missing/invalid samples.
%
%   Coordinate Transformation:
%     Simscape Z-up [x; y; z] is formed from capture Y-up [x_c; y_c; z_c] as:
%       x_si =  x_c * scale
%       y_si = -z_c * scale
%       z_si =  y_c * scale
%
%   See also GS3DX_CAPTURE_MARKERS.

    % Validate points shape and type
    if ~isnumeric(points) || ~isreal(points) || ndims(points) > 3 || size(points, 1) ~= 4
        error('gs3dx:capture_points:badPoints', ...
            'POINTS must be a real numeric 4 x N x T array.');
    end
    n_markers = size(points, 2);
    n_frames = size(points, 3);
    if n_markers < 1 || n_frames < 1
        error('gs3dx:capture_points:badPoints', ...
            'POINTS must have at least 1 marker and 1 frame.');
    end

    % Validate residuals shape and type
    if ~isnumeric(residuals) || ~isreal(residuals) || ndims(residuals) > 3 || size(residuals, 1) ~= 1
        error('gs3dx:capture_points:badResiduals', ...
            'RESIDUALS must be a real numeric 1 x N x T array.');
    end
    if size(residuals, 2) ~= n_markers || size(residuals, 3) ~= n_frames
        error('gs3dx:capture_points:dimensionMismatch', ...
            'RESIDUALS dimensions (1 x %d x %d) must match POINTS marker and frame dimensions (%d x %d).', ...
            size(residuals, 2), size(residuals, 3), n_markers, n_frames);
    end

    % Validate units (scalar m, mm, cm)
    if ischar(units) && (isrow(units) || isempty(units))
        u_str = strtrim(units);
    elseif isstring(units) && isscalar(units)
        u_str = strtrim(char(units));
    else
        error('gs3dx:capture_points:invalidUnits', ...
            'UNITS must be a scalar string or char vector (''m'', ''mm'', or ''cm'').');
    end

    switch lower(u_str)
        case 'm'
            scale = 1.0;
        case 'mm'
            scale = 1e-3;
        case 'cm'
            scale = 1e-2;
        otherwise
            error('gs3dx:capture_points:unsupportedUnits', ...
                'Unsupported units ''%s''; only ''m'', ''mm'', and ''cm'' are supported.', u_str);
    end

    % Validate optional observed_mask
    if nargin < 4
        has_obs_mask = false;
    elseif isempty(observed_mask)
        has_obs_mask = false;
        if ~((islogical(observed_mask) || isa(observed_mask, 'double')) ...
                && isreal(observed_mask) && isequal(size(observed_mask), [0, 0]))
            error('gs3dx:capture_points:badObservedMask', ...
                'Empty OBSERVED_MASK must be the double [] or logical 0 x 0 sentinel.');
        end
    else
        has_obs_mask = true;
        if ~islogical(observed_mask)
            error('gs3dx:capture_points:badObservedMask', ...
                'OBSERVED_MASK must be a logical array.');
        end
        if ndims(observed_mask) > 3
            error('gs3dx:capture_points:badObservedMask', ...
                'OBSERVED_MASK cannot have more than 3 dimensions.');
        end

        % Singleton handling
        if n_frames == 1 && n_markers > 1 && isequal(size(observed_mask), [n_markers, 1])
            obs_mask = reshape(observed_mask, [1, n_markers, 1]);
        elseif n_markers == 1 && n_frames > 1 && (isequal(size(observed_mask), [1, n_frames]) || isequal(size(observed_mask), [n_frames, 1]))
            obs_mask = reshape(observed_mask, [1, 1, n_frames]);
        elseif n_markers == 1 && n_frames == 1 && isscalar(observed_mask)
            obs_mask = reshape(observed_mask, [1, 1, 1]);
        else
            obs_mask = observed_mask;
        end

        if size(obs_mask, 1) ~= 1 || size(obs_mask, 2) ~= n_markers || size(obs_mask, 3) ~= n_frames
            error('gs3dx:capture_points:dimensionMismatch', ...
                'OBSERVED_MASK dimensions (1 x %d x %d) must match POINTS marker and frame dimensions (%d x %d).', ...
                size(obs_mask, 2), size(obs_mask, 3), n_markers, n_frames);
        end
    end

    % Ensure working precision as double
    pts = double(points);
    res = double(residuals);

    % Compute markerwise missing mask (1 x N x T)
    % Invalid if negative residual, nonfinite residual, nonfinite XYZ coordinates,
    % or unobserved per observed_mask (~obs_mask).
    % Row 4 is homogeneous 1 and is never checked as a residual or coordinate.
    bad_res = (res < 0) | ~isfinite(res);
    bad_xyz = any(~isfinite(pts(1:3, :, :)), 1);
    if has_obs_mask
        missing_mask = bad_res | bad_xyz | ~obs_mask;
    else
        missing_mask = bad_res | bad_xyz;
    end

    % Coordinate conversion: Y-up (x, y, z) -> Simscape Z-up (x, -z, y)
    x_si = pts(1, :, :) * scale;
    y_si = -pts(3, :, :) * scale;
    z_si = pts(2, :, :) * scale;
    points_si = cat(1, x_si, y_si, z_si);

    % Mask invalid samples to NaN across all 3 coordinates
    mask_3d = repmat(missing_mask, [3, 1, 1]);
    points_si(mask_3d) = NaN;

    % Postcondition contract assertions
    assert(isreal(points_si) && isnumeric(points_si), 'Postcondition: POINTS_SI must be real numeric.');
    assert(size(points_si, 1) == 3 && size(points_si, 2) == n_markers && size(points_si, 3) == n_frames, ...
        'Postcondition: POINTS_SI shape must be 3 x N x T.');
    assert(islogical(missing_mask), 'Postcondition: MISSING_MASK must be logical.');
    assert(size(missing_mask, 1) == 1 && size(missing_mask, 2) == n_markers && size(missing_mask, 3) == n_frames, ...
        'Postcondition: MISSING_MASK shape must be 1 x N x T.');
end
