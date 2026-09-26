function E = gs3dx_xyz_rate_matrix(b, c)
%GS3DX_XYZ_RATE_MATRIX  Euler-rate to follower-frame angular velocity map.
%
%   E = GS3DX_XYZ_RATE_MATRIX(B, C) returns the 3x3 matrix with
%   omega_F = E * [da; db; dc] for the intrinsic X-Y-Z sequence
%   R = Rx(a) * Ry(b) * Rz(c) that a Gimbal Joint uses (angles in radians;
%   omega_F is the follower's angular velocity relative to the base,
%   resolved in the follower frame).  E does not depend on a.
%
%   E is singular where cos(b) = 0 (gimbal lock).  Code-generation
%   compatible: called from the MATLAB Function block in GS3DX_KDS_Spherical.

    sb = sin(b); cb = cos(b); sc = sin(c); cc = cos(c);
    E = [cb * cc,  sc, 0; ...
        -cb * sc,  cc, 0; ...
         sb,       0,  1];
end
