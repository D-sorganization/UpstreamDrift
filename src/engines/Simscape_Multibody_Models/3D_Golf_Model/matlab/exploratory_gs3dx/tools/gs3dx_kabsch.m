function [R, c] = gs3dx_kabsch(M0, c0, M)
%GS3DX_KABSCH  Least-squares rigid rotation and centroid (#11160).
%
%   [R, C] = GS3DX_KABSCH(M0, C0, M) with M - C = R * (M0 - C0).

    arguments
        M0 (3,6) double
        c0 (3,1) double
        M (3,6) double
    end
    c = mean(M, 2);
    [U, ~, V] = svd((M0 - c0) * (M - c).');
    R = V * diag([1 1 det(V * U.')]) * U.';
end
