function [R, gaps_out, meta] = gs3dx_foot_marker_frame(ankle, toe_in, toe_out, gaps, opts)
%GS3DX_FOOT_MARKER_FRAME  Calibrated 3D foot orientation from motion capture markers (#10979, #11161).
%
%   [R, GAPS_OUT, META] = GS3DX_FOOT_MARKER_FRAME(ANKLE, TOE_IN, TOE_OUT, GAPS, OPTS)
%   constructs an anatomical 3D foot orientation frame for each frame f in 1..N
%   using skin/shoe motion capture markers.
%
%   Scientific Rationale:
%     The raw vector from ankle marker to toe marker is NOT the shoe +x forward
%     axis: surface ankle markers sit higher than toe markers, introducing a
%     downward pitch bias when using an uncalibrated ankle-to-toe direction
%     vector for a flat shoe.
%     Furthermore, a forward direction vector alone cannot constrain the foot's
%     roll degree of freedom (an upside-down sole has the same forward axis).
%     This helper constructs a full 3D marker triad F, calibrates it against the
%     address stance via SVD proper mean rotation, and rotates by the address yaw
%     measured in the horizontal plane:
%
%       R(f) = F(f) * F_address' * Rz(address_yaw)
%
%   Triad Construction:
%     For each frame f:
%       toe_mean = (toe_in(:, f) + toe_out(:, f)) / 2
%       x = normalized(toe_mean - ankle(:, f))
%       z = normalized(cross(x, toe_in(:, f) - toe_out(:, f)))
%       y = cross(z, x)
%       F(:, :, f) = [x, y, z]   (proper right-handed orthonormal triad in SO(3))
%
%   Address Calibration:
%     F_address is the SVD proper mean of F over valid (gap=false) address frames:
%       [U, ~, V] = svd(sum(F(:, :, address_valid), 3))
%       F_address = U * diag([1, 1, det(U * V')]) * V'
%     R_address = Rz(address_yaw)
%
%   Modeling Assumptions & Boundaries:
%     1. Flat Sole at Address is an ASSUMPTION: The golfer is assumed to have their
%        shoe soles flat on the ground at address (+z is vertical [0; 0; 1]).
%        This is a kinematic calibration assumption, NOT ground force or contact
%        equilibrium evidence.
%     2. Horizontal Yaw: Address yaw is measured or defined in the horizontal plane.
%     3. Marker-Height Slope Calibration: The downward pitch of the marker triad
%        relative to the true sole is calibrated out by F_address'.
%     4. Fail-Closed Validation: Measured frames (gap=false) must be real, finite,
%        and non-degenerate (toe-ankle separation >= 1e-4 m, toe spread >= 1e-4 m,
%        non-collinear). Only explicit gaps permit missing or unmeasured frames.
%
%   Inputs:
%     ankle    (3, N) double coordinates (m)
%     toe_in   (3, N) double coordinates (m)
%     toe_out  (3, N) double coordinates (m)
%     gaps     (1, N) logical, true where markers are gap-filled/unmeasured
%     opts.address_frames   (1, M) double/integer address frame indices (default 1)
%     opts.address_yaw_deg  (1, 1) double address yaw angle (deg, default horizontal toe-ankle)
%
%   Outputs:
%     R         (3, 3, N) double SO(3) rotation matrices per frame (NaN where missing)
%     gaps_out  (1, N) logical gap mask
%     meta      struct containing triad F, F_address, R_address, yaw, and assumptions

    arguments
        ankle (3,:) double
        toe_in (3,:) double
        toe_out (3,:) double
        gaps (1,:) logical = false(1, size(ankle, 2))
        opts.address_frames double = 1
        opts.address_yaw_deg double = []
    end

    n = size(ankle, 2);
    if size(toe_in, 1) ~= 3 || size(toe_in, 2) ~= n || ...
       size(toe_out, 1) ~= 3 || size(toe_out, 2) ~= n
        error('gs3dx:foot_marker_frame:invalid_dimension', ...
            'ankle, toe_in, and toe_out must all have size 3xN (found ankle 3x%d, toe_in %dx%d, toe_out %dx%d)', ...
            n, size(toe_in, 1), size(toe_in, 2), size(toe_out, 1), size(toe_out, 2));
    end
    if numel(gaps) ~= n
        error('gs3dx:foot_marker_frame:invalid_dimension', ...
            'gaps length (%d) must match number of frames (%d)', numel(gaps), n);
    end
    gaps = gaps(:).';

    if isempty(opts.address_frames) || ~isvector(opts.address_frames) || ~isreal(opts.address_frames) || ...
       ~all(isfinite(opts.address_frames), 'all') || any(opts.address_frames ~= round(opts.address_frames), 'all') || ...
       any(opts.address_frames < 1 | opts.address_frames > n, 'all')
        error('gs3dx:foot_marker_frame:invalid_address', ...
            'address_frames indices must be non-empty real finite positive integers within 1..N (1..%d)', n);
    end
    addr = opts.address_frames(:).';

    toe_mean = (toe_in + toe_out) / 2;
    F = nan(3, 3, n);

    for f = 1:n
        if gaps(f)
            % Only explicit gaps permit non-finite, complex, or degenerate markers
            if isreal(ankle(:, f)) && isreal(toe_in(:, f)) && isreal(toe_out(:, f)) && ...
               all(isfinite(ankle(:, f))) && all(isfinite(toe_in(:, f))) && all(isfinite(toe_out(:, f)))
                x = toe_mean(:, f) - ankle(:, f);
                nx = norm(x);
                v_toe = toe_in(:, f) - toe_out(:, f);
                nv = norm(v_toe);
                if nx >= 1e-4 && nv >= 1e-4
                    x = x / nx;
                    z = cross(x, v_toe);
                    nz = norm(z);
                    if nz >= 1e-4
                        z = z / nz;
                        y = cross(z, x);
                        F(:, :, f) = [x, y, z];
                    end
                end
            end
            continue;
        end

        % Measured frame: strictly validate finite real coordinates
        a = ankle(:, f);
        ti = toe_in(:, f);
        to = toe_out(:, f);
        if ~isreal(a) || ~isreal(ti) || ~isreal(to) || ...
           ~all(isfinite(a)) || ~all(isfinite(ti)) || ~all(isfinite(to))
            error('gs3dx:foot_marker_frame:invalid_measurement', ...
                'Non-finite or complex marker coordinates at frame %d with gap=false', f);
        end

        x = toe_mean(:, f) - a;
        nx = norm(x);
        if nx < 1e-4
            error('gs3dx:foot_marker_frame:degenerate_triad', ...
                'Degenerate ankle-to-toe distance (< 1e-4 m) at frame %d', f);
        end
        x = x / nx;

        v_toe = ti - to;
        nv = norm(v_toe);
        if nv < 1e-4
            error('gs3dx:foot_marker_frame:degenerate_triad', ...
                'Degenerate ToeIn-to-ToeOut distance (< 1e-4 m) at frame %d', f);
        end

        z = cross(x, v_toe);
        nz = norm(z);
        if nz < 1e-4
            error('gs3dx:foot_marker_frame:degenerate_triad', ...
                'Collinear foot marker triad at frame %d', f);
        end
        z = z / nz;

        y = cross(z, x);
        F(:, :, f) = [x, y, z];
    end

    % Locate valid measured address frames
    ok_addr = addr(~gaps(addr));
    if isempty(ok_addr)
        error('gs3dx:foot_marker_frame:no_valid_address', ...
            'No valid measured address frame without gap in address_frames');
    end

    % SVD proper mean rotation over address frames
    sumF = sum(F(:, :, ok_addr), 3);
    [U, S, V] = svd(sumF);
    s_diag = diag(S);
    d_uv = det(U * V.');
    if s_diag(2) < 1e-4 || (d_uv < 0 && (s_diag(2) - s_diag(3) < 1e-4))
        error('gs3dx:foot_marker_frame:degenerate_mean', ...
            'Degenerate or rank-deficient address marker triads cannot define a unique mean orientation');
    end
    F0 = U * diag([1, 1, d_uv]) * V.';

    % Address yaw in horizontal plane
    if isempty(opts.address_yaw_deg)
        f0 = ok_addr(1);
        dx = toe_mean(1, f0) - ankle(1, f0);
        dy = toe_mean(2, f0) - ankle(2, f0);
        if hypot(dx, dy) < 1e-4
            error('gs3dx:foot_marker_frame:degenerate_triad', ...
                'Degenerate horizontal ankle-to-toe vector at address frame %d', f0);
        end
        yaw_deg = atan2d(dy, dx);
    else
        if ~isscalar(opts.address_yaw_deg) || ~isreal(opts.address_yaw_deg) || ~isfinite(opts.address_yaw_deg)
            error('gs3dx:foot_marker_frame:invalid_yaw', ...
                'address_yaw_deg must be empty or a real finite scalar');
        end
        yaw_deg = opts.address_yaw_deg;
    end

    % R_address: rotation about vertical +z by yaw_deg
    R0 = [cosd(yaw_deg), -sind(yaw_deg), 0; sind(yaw_deg), cosd(yaw_deg), 0; 0, 0, 1];

    % Map every frame: R(f) = F(f) * F0' * R0
    R = nan(3, 3, n);
    gaps_out = gaps;
    for f = 1:n
        if isreal(F(:, :, f)) && all(isfinite(F(:, :, f)), 'all')
            R(:, :, f) = F(:, :, f) * F0.' * R0;
        else
            gaps_out(f) = true;
        end
    end

    if nargout > 2
        meta = struct( ...
            'F', F, ...
            'F_address', F0, ...
            'R_address', R0, ...
            'address_yaw_deg', yaw_deg, ...
            'address_frames', addr, ...
            'valid_address_frames', ok_addr, ...
            'assumption', 'Flat sole at address is an ASSUMPTION, not force/contact evidence; yaw measured in horizontal plane.' ...
        );
    end
end
