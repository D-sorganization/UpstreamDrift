function q = gs3dx_rebase_turns(q)
%GS3DX_REBASE_TURNS  Shift reference rows by whole turns (#11173).
%
%   Q = GS3DX_REBASE_TURNS(Q) with Q (RxN, finite, deg) rebases each row
%   independently: k = round(median(row)/360), then row -= 360*k so the
%   median lies in (-180, 180] while the row's span is unchanged.
%
%   Preconditions: Q finite; at least one column.
%   Postcondition: size unchanged; each row's max-min span unchanged within
%   1e-9; each median in (-180, 180].

    arguments
        q (:, :) double {mustBeFinite}
    end

    if size(q, 2) < 1
        error('gs3dx:rate', 'Need at least one column');
    end

    q0 = q;
    for r = 1:size(q, 1)
        med = median(q(r, :));
        k = round(med / 360);
        q(r, :) = q(r, :) - 360 * k;
    end

    assert(isequal(size(q), size(q0)), 'gs3dx:rate', 'Postcondition failed: size changed');
    for r = 1:size(q, 1)
        span0 = max(q0(r, :)) - min(q0(r, :));
        span1 = max(q(r, :)) - min(q(r, :));
        assert(abs(span1 - span0) <= 1e-9, 'gs3dx:rate', ...
            'Postcondition failed: row %d span changed', r);
        med = median(q(r, :));
        assert(med > -180 && med <= 180, 'gs3dx:rate', ...
            'Postcondition failed: row %d median not in (-180, 180]', r);
    end
end
