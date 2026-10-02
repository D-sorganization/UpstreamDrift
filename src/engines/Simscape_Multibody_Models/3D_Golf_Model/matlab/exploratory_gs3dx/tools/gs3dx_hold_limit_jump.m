function q = gs3dx_hold_limit_jump(q, lo, hi)
%GS3DX_HOLD_LIMIT_JUMP  Remove adjacent samples on opposite ROM limits (#11173).
%
%   Q = GS3DX_HOLD_LIMIT_JUMP(Q, LO, HI) walks each row of Q (RxN, finite)
%   left to right.  When sample j is within 1e-6 of one limit and sample
%   j-1 is within 1e-6 of the other, sample j is set equal to sample j-1.
%   All other samples are unchanged.
%
%   Preconditions: Q, LO and HI finite; HI > LO.
%   Postcondition: size unchanged; every sample that was inside [LO, HI] stays
%   inside within 1e-9; no adjacent pair has one sample on LO and the other
%   on HI (1e-6).

    arguments
        q (:, :) double {mustBeFinite}
        lo (1, 1) double {mustBeFinite}
        hi (1, 1) double {mustBeFinite}
    end

    if hi <= lo
        error('gs3dx:rate', 'hi must exceed lo');
    end

    tol = 1e-6;
    q0 = q;
    q_out = q;
    at_lo = @(x) abs(x - lo) <= tol;
    at_hi = @(x) abs(x - hi) <= tol;
    for r = 1:size(q, 1)
        for j = 2:size(q, 2)
            if (at_lo(q_out(r, j)) && at_hi(q_out(r, j - 1))) || ...
                    (at_hi(q_out(r, j)) && at_lo(q_out(r, j - 1)))
                q_out(r, j) = q_out(r, j - 1);
            end
        end
    end

    assert(isequal(size(q_out), size(q0)), 'gs3dx:rate', 'Postcondition failed: size changed');
    inside0 = q0 >= lo & q0 <= hi;
    assert(all(q_out(inside0) >= lo - 1e-9 & q_out(inside0) <= hi + 1e-9), ...
        'gs3dx:rate', 'Postcondition failed: an in-range sample left [lo, hi]');
    for r = 1:size(q_out, 1)
        for j = 2:size(q_out, 2)
            assert(~((abs(q_out(r, j) - lo) <= tol && abs(q_out(r, j - 1) - hi) <= tol) || ...
                (abs(q_out(r, j) - hi) <= tol && abs(q_out(r, j - 1) - lo) <= tol)), ...
                'gs3dx:rate', 'Postcondition failed: adjacent lo/hi pair remains');
        end
    end

    q = q_out;
end
