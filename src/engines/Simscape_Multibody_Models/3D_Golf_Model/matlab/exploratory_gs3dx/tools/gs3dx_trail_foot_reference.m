function feet = gs3dx_trail_foot_reference(feet, h)
%GS3DX_TRAIL_FOOT_REFERENCE  Lower trail-foot z reference by preload height.
%
%   FEET = GS3DX_TRAIL_FOOT_REFERENCE(FEET, H) returns 6xN BalanceFootRef-style
%   foot references.  H is 1xN metres from GS3DX_TRAIL_PRELOAD.  Row 6 (trail
%   foot z) becomes the input row 6 minus H; rows 1:5 are unchanged.
%
%   Error id gs3dx:preload.

    assert(size(feet, 1) == 6, 'gs3dx:preload', 'FEET must be 6xN');
    n = size(feet, 2);
    h = h(:).';
    assert(numel(h) == n, 'gs3dx:preload', 'H must match FEET column count');
    assert(all(isfinite(feet(:))), 'gs3dx:preload', 'FEET must be finite');
    assert(all(isfinite(h)), 'gs3dx:preload', 'H must be finite');
    assert(all(h >= -1e-12), 'gs3dx:preload', 'H must be non-negative');

    lead = feet(1:5, :);
    zIn = feet(6, :);
    feet(6, :) = zIn - h;

    assert(max(abs(feet(6, :) - (zIn - h)), [], 'all') <= 1e-12, 'gs3dx:preload', ...
        'Trail foot z must equal input minus H');
    assert(max(abs(feet(1:5, :) - lead), [], 'all') <= 1e-12, 'gs3dx:preload', ...
        'Rows 1:5 must be unchanged');
end
