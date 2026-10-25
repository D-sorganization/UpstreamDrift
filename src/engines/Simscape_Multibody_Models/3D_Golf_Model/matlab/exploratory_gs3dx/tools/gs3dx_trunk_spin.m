function h = gs3dx_trunk_spin(hip_L, hip_R, shoulder_L, shoulder_R, mass, gyration, dt)
%GS3DX_TRUNK_SPIN  Trunk spin angular momentum from joint centres (#11173).
%
%   H = GS3DX_TRUNK_SPIN(HIP_L, HIP_R, SHOULDER_L, SHOULDER_R, MASS,
%   GYRATION, DT) estimates trunk rigid-body spin about principal axes.
%   Hip and shoulder points are 3xN in one frame.  MASS is trunk mass (kg).
%   GYRATION is 1x3 de Leva radii [sagittal transverse longitudinal]
%   (gs3dx_anthropometry .gyration.trunk).  The caller filters joint
%   centres; this function filters nothing.
%
%   Postcondition: H is 3xN and finite.

    arguments
        hip_L (3, :) double
        hip_R (3, :) double
        shoulder_L (3, :) double
        shoulder_R (3, :) double
        mass (1, 1) double {mustBePositive, mustBeFinite}
        gyration (1, 3) double {mustBePositive, mustBeLessThan(gyration, 1)}
        dt (1, 1) double {mustBePositive, mustBeFinite}
    end

    n = size(hip_L, 2);
    assert(isequal(size(hip_R), [3, n]) && isequal(size(shoulder_L), [3, n]) ...
        && isequal(size(shoulder_R), [3, n]), 'gs3dx:spin', ...
        'Hip and shoulder trajectories must be 3xN with a common N');

    hip_mid = (hip_L + hip_R) / 2;
    shoulder_mid = (shoulder_L + shoulder_R) / 2;
    R = zeros(3, 3, n);

    for k = 1:n
        seg = shoulder_mid(:, k) - hip_mid(:, k);
        len = norm(seg);
        if len < 0.01
            error('gs3dx:spin', 'Hip-shoulder distance %.4g m at frame %d is under 1 cm', len, k);
        end
        u = seg / len;
        y0 = unit_vec(hip_R(:, k) - hip_L(:, k), k, 'Hip width');
        v = unit_vec(y0 - u * dot(y0, u), k, 'Trunk lateral axis');
        w_axis = cross(u, v);
        if det([w_axis, v, u]) < 0
            w_axis = -w_axis;
        end
        R(:, :, k) = [w_axis, v, u];
    end

    omega = gs3dx_frame_rate(R, dt);
    h = zeros(3, n);
    for k = 1:n
        len = norm(shoulder_mid(:, k) - hip_mid(:, k));
        [~, ~, I] = gs3dx_segment_inertia(mass, len, 0.5, gyration);
        h(:, k) = R(:, :, k) * (I(:) .* omega(:, k));
    end

    assert(all(isfinite(h(:))), 'gs3dx:spin', 'Postcondition: H must be finite');
end

function u = unit_vec(v, frame, label)
    nrm = norm(v);
    if nrm < 1e-9
        error('gs3dx:spin', '%s too short at frame %d', label, frame);
    end
    u = v / nrm;
end
