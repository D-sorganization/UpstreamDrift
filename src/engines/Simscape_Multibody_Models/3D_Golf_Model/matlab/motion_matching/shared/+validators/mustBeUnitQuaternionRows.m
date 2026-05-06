function mustBeUnitQuaternionRows(Q, tol)
%MUSTBEUNITQUATERNIONROWS Validate that Q is an Nx4 array of unit quaternions.
%
%   mustBeUnitQuaternionRows(Q) checks that each row of Q has 2-norm equal
%   to 1 within absolute tolerance 1e-6 (per CLUB_IK_SPEC.md § Validation).
%
%   mustBeUnitQuaternionRows(Q, TOL) overrides the default tolerance.
%
%   Error identifiers:
%     validator:badShape     - Q is not an Nx4 real numeric array
%     validator:notFinite    - Q contains NaN or Inf
%     validator:notUnitNorm  - one or more rows have ||q|| outside [1-tol, 1+tol]
    arguments
        Q
        tol (1,1) double {mustBePositive} = 1e-6
    end
    if ~isnumeric(Q) || ~isreal(Q) || ndims(Q) ~= 2 || size(Q, 2) ~= 4 || isempty(Q)
        error("validator:badShape", ...
              "Quaternion array must be a non-empty Nx4 real numeric matrix.");
    end
    if any(~isfinite(Q(:)))
        error("validator:notFinite", ...
              "Quaternion array contains NaN or Inf.");
    end
    norms = sqrt(sum(Q .^ 2, 2));
    if any(abs(norms - 1) > tol)
        worst = max(abs(norms - 1));
        error("validator:notUnitNorm", ...
              "Quaternion rows must have unit norm within %.2g; worst deviation %.3g.", ...
              tol, worst);
    end
end
