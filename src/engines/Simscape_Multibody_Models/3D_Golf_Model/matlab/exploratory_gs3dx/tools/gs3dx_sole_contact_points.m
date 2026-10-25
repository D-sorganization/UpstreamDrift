function [centres, points] = gs3dx_sole_contact_points(geo, feet)
%GS3DX_SOLE_CONTACT_POINTS  World sole-sphere centres and ground contact points (#11709).
%
%   [CENTRES, POINTS] = GS3DX_SOLE_CONTACT_POINTS(GEO, FEET) places the six
%   sole spheres (heel, toe-in, toe-out per foot) in world coordinates from
%   each foot's ankle pose, using the same per-foot corner offsets
%   GS3DX_BUILD_CONTACT uses to build the contacts:
%     off = [x; side*y*FootContactWidth/2; -AnkleHeight+FootContactRadius]
%   in the ankle-distal foot frame, with
%     x = -FootHeelOffset*FootLength            (heel)
%     x = (1-FootHeelOffset)*FootLength         (toe-in, toe-out)
%     y = 0 (heel), -1 (toe-in, inside), +1 (toe-out, outside)
%   GEO is GS3DX_CONTACT_GEOMETRY's struct.  FEET is a struct with fields
%   L and R, each with .p (3xN, world ankle position) and .R (3x3xN, world
%   ankle rotation, as logged by the ankle joint's Rotation_Transform).
%
%   CENTRES is 3 x 6 x N (sphere centres, world); POINTS is CENTRES shifted
%   down by one sphere radius along world +Z (the sphere's lowest point,
%   the ground contact point used for the GRF fixture receipt).  Column
%   order: L Heel, L ToeIn, L ToeOut, R Heel, R ToeIn, R ToeOut -- matches
%   FootContactForces (GS3DX_BUILD_CONTACT) and
%   src.engines.simscape.ground_contact.FOOT_CONTACTS (Python).
%
%   Shared by EXPORT_GS3DX_GRF_FIXTURE and GS3DX_RIGID_STANCE: the rigid
%   stance proxy's six fixed-in-body sphere offsets are this function's
%   CENTRES at the probe pose, expressed in the body frame.
%
%   Preconditions: FEET.L.p and FEET.R.p are 3xN; FEET.L.R and FEET.R.R are
%   3x3xN with the same N as FEET.L.p/FEET.R.p.

    arguments
        geo (1,1) struct
        feet (1,1) struct
    end
    up = [0; 0; 1];
    n_t = size(feet.L.p, 2);
    assert(size(feet.R.p, 2) == n_t, 'gs3dx:sole_contact_points', ...
        'Precondition: L (%d) and R (%d) sample counts differ', n_t, size(feet.R.p, 2));
    % size(X, 3) is 1 for a plain 3x3 matrix (n_t == 1), so this also accepts
    % the trailing dimension MATLAB drops when a 3x3xN array is indexed down
    % to one page.
    ok = @(R) size(R, 1) == 3 && size(R, 2) == 3 && size(R, 3) == n_t && ndims(R) <= 3;
    assert(ok(feet.L.R) && ok(feet.R.R), 'gs3dx:sole_contact_points', ...
        'Precondition: FEET.L.R/FEET.R.R must be 3x3x%d', n_t);
    corners = struct('name', {'Heel', 'ToeIn', 'ToeOut'}, ...
        'x', {-geo.FootHeelOffset * geo.FootLength, ...
              (1 - geo.FootHeelOffset) * geo.FootLength, ...
              (1 - geo.FootHeelOffset) * geo.FootLength}, ...
        'y', {0, -1, 1});   % -1 = inside (toward the other foot)
    centres = zeros(3, 6, n_t);
    col = 0;
    for side = struct('P', {'L', 'R'}, 'sign', {1, -1})
        foot = feet.(side.P);
        for k = 1:numel(corners)
            col = col + 1;
            off = [corners(k).x; side.sign * corners(k).y * geo.FootContactWidth / 2; ...
                -geo.AnkleHeight + geo.FootContactRadius];
            centres(:, col, :) = reshape(foot.p + squeeze(pagemtimes(foot.R, off)), 3, 1, []);
        end
    end
    points = centres - geo.FootContactRadius * up;
end
