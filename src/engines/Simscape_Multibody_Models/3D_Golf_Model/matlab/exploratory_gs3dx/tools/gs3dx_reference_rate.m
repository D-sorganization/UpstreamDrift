function qd = gs3dx_reference_rate(t, q)
%GS3DX_REFERENCE_RATE  Time derivative of reference angles on a time base (#11173).
%
%   QD = GS3DX_REFERENCE_RATE(T, Q) with T (1xN, s, strictly increasing) and
%   Q (RxN, same angle unit as the references) returns QD (RxN, unit/s) from
%   gradient(Q, T) along the frame axis, matching gs3dx_leg_reference's
%   gradient(q, 1/rate) on uniform samples.
%
%   Preconditions: T and Q finite; numel(T) == size(Q, 2); N >= 2; T strictly
%   increasing.
%   Postcondition: size(QD) == size(Q); QD finite.

    arguments
        t (1, :) double {mustBeFinite}
        q (:, :) double {mustBeFinite}
    end

    n = numel(t);
    if n < 2
        error('gs3dx:rate', 'Need at least two time samples');
    end
    if size(q, 2) ~= n
        error('gs3dx:rate', 'numel(t) must equal size(q, 2)');
    end
    if any(diff(t) <= 0)
        error('gs3dx:rate', 't must be strictly increasing');
    end

    qd = zeros(size(q));
    for r = 1:size(q, 1)
        qd(r, :) = gradient(q(r, :), t);
    end
    assert(isequal(size(qd), size(q)) && all(isfinite(qd), 'all'), 'gs3dx:rate', ...
        'Postcondition failed: qd must match q and be finite');
end
