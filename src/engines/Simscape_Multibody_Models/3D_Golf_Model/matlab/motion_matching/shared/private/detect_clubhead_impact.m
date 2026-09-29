function idx = detect_clubhead_impact(t, clubhead)
%DETECT_CLUBHEAD_IMPACT  Index of max ||d r_clubhead/dt|| (5-point stencil).
%
%   IDX = DETECT_CLUBHEAD_IMPACT(T, CLUBHEAD) returns the 1-based index into
%   T at which the magnitude of the time derivative of CLUBHEAD is maximal.
%   The derivative uses a 5-point central-difference stencil after
%   resampling to a uniform grid (per CLUB_IK_SPEC.md §"Time alignment");
%   for fewer than 5 frames it falls back to MATLAB's GRADIENT.
%
%   Inputs:
%     T         (M,1) double  monotonic non-decreasing time vector (s)
%     CLUBHEAD  (M,3) double  clubhead position in metres
%
%   Output:
%     IDX       (1,1) double  1 <= IDX <= M
%
%   This helper is the single source of truth for impact detection across
%   the motion_matching tree (DRY).  align_to_simulation_grid.m and
%   synthesize_target_from_coefficients.m both call it.
    arguments
        t        (:,1) double {mustBeReal, mustBeNonempty}
        clubhead (:,3) double {mustBeReal}
    end

    M = numel(t);
    assert(size(clubhead, 1) == M, ...
        "detect_clubhead_impact:rowMismatch", ...
        "T and CLUBHEAD must have the same number of rows");

    if M < 5
        v = zeros(M, 3);
        for d = 1:3
            v(:, d) = gradient(clubhead(:, d), t);
        end
    else
        % Uniform resample for velocity computation
        tu = linspace(t(1), t(end), M).';
        cu = interp1(t, clubhead, tu, "linear");
        h  = tu(2) - tu(1);
        vu = zeros(M, 3);
        for d = 1:3
            col = cu(:, d);
            vu(3:M-2, d) = (-col(5:M) + 8*col(4:M-1) - 8*col(2:M-3) + col(1:M-4)) / (12*h);
            vu(1, d)     = (col(2) - col(1)) / h;
            vu(2, d)     = (col(3) - col(1)) / (2*h);
            vu(M-1, d)   = (col(M) - col(M-2)) / (2*h);
            vu(M, d)     = (col(M) - col(M-1)) / h;
        end
        v = interp1(tu, vu, t, "linear", "extrap");
    end
    speed = sqrt(sum(v.^2, 2));
    [~, idx] = max(speed);

    assert(idx >= 1 && idx <= M, ...
        "detect_clubhead_impact:badIndex", ...
        "Postcondition: 1 <= idx <= M");
end
