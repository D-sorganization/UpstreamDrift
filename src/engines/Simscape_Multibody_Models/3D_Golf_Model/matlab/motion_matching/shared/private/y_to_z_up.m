function out = y_to_z_up(points)
%Y_TO_Z_UP  Convert Y-up positions/rotations to right-handed Z-up.
%
%   OUT = Y_TO_Z_UP(POINTS) applies the right-handed swap
%       (x, y, z) -> (x, -z, y)
%   which is the rotation
%       R = [1 0 0; 0 0 -1; 0 1 0],   det(R) = +1.
%
%   Accepts:
%     - Nx3 double      (positions)              -> Nx3
%     - 3x3 double      (rotation matrix)        -> 3x3 (R_swap * R)
%     - 3x3xN double    (stack of rotations)     -> 3x3xN
%
%   Used to bring Vicon Y-up cluster-marker mocap into the Simscape Z-up world
%   frame.  Sign convention chosen (rather than (x,z,-y)) so that ground-
%   plane Y heights become positive Z heights, matching the simulation
%   convention where Z is "up off the tee".
    arguments
        points double {mustBeFinite}
    end

    R = [1 0 0; 0 0 -1; 0 1 0];

    sz = size(points);
    if numel(sz) == 2 && sz(2) == 3
        % Nx3 positions: row vectors.
        out = points * R.';
    elseif isequal(sz, [3 3])
        out = R * points;
    elseif numel(sz) == 3 && sz(1) == 3 && sz(2) == 3
        out = zeros(size(points));
        for k = 1:size(points, 3)
            out(:, :, k) = R * points(:, :, k);
        end
    else
        error("y_to_z_up:badShape", ...
              "Input must be Nx3, 3x3, or 3x3xN; got size %s", mat2str(sz));
    end

    assert(abs(det(R) - 1) < 1e-12, "Postcondition: det(R) == +1");
end
