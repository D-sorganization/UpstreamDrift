function quats = rotmat_to_quaternion(R)
%ROTMAT_TO_QUATERNION  Convert 3x3 rotation matrix/matrices to unit quaternions.
%
%   QUATS = ROTMAT_TO_QUATERNION(R) returns Nx4 [w x y z] unit quaternions for
%   the input rotations.  R may be a 3x3 single rotation or a 3x3xN stack.
%   The sign is canonicalised so that q(1) (== w) >= 0, eliminating the
%   q <-> -q ambiguity at the source as required by CLUB_IK_SPEC.md.
%
%   Algorithm: Shepperd's method (numerically stable across all orientations).
%
%   Preconditions:
%     - R is 3x3 or 3x3xN double, each slice approximately orthogonal.
%
%   Postconditions:
%     - Each row of QUATS is unit-norm to within 1e-9.
%     - QUATS(:,1) >= 0 for every row.
    arguments
        R (3,3,:) double {mustBeFinite}
    end

    n = size(R, 3);
    quats = zeros(n, 4);

    for k = 1:n
        M = R(:,:,k);
        tr = M(1,1) + M(2,2) + M(3,3);

        if tr > 0
            S = 2 * sqrt(tr + 1.0);
            w = 0.25 * S;
            x = (M(3,2) - M(2,3)) / S;
            y = (M(1,3) - M(3,1)) / S;
            z = (M(2,1) - M(1,2)) / S;
        elseif (M(1,1) > M(2,2)) && (M(1,1) > M(3,3))
            S = 2 * sqrt(1.0 + M(1,1) - M(2,2) - M(3,3));
            w = (M(3,2) - M(2,3)) / S;
            x = 0.25 * S;
            y = (M(1,2) + M(2,1)) / S;
            z = (M(1,3) + M(3,1)) / S;
        elseif M(2,2) > M(3,3)
            S = 2 * sqrt(1.0 + M(2,2) - M(1,1) - M(3,3));
            w = (M(1,3) - M(3,1)) / S;
            x = (M(1,2) + M(2,1)) / S;
            y = 0.25 * S;
            z = (M(2,3) + M(3,2)) / S;
        else
            S = 2 * sqrt(1.0 + M(3,3) - M(1,1) - M(2,2));
            w = (M(2,1) - M(1,2)) / S;
            x = (M(1,3) + M(3,1)) / S;
            y = (M(2,3) + M(3,2)) / S;
            z = 0.25 * S;
        end

        q = [w, x, y, z];
        q = q / norm(q);
        if q(1) < 0
            q = -q;
        end
        quats(k, :) = q;
    end

    norms = sqrt(sum(quats.^2, 2));
    assert(all(abs(norms - 1) < 1e-9), ...
        "Postcondition: all output quaternions must be unit-norm");
    assert(all(quats(:,1) >= 0), ...
        "Postcondition: q(1) (w) must be non-negative for all rows");
end
