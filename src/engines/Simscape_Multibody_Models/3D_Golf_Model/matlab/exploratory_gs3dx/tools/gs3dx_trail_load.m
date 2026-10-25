function feet = gs3dx_trail_load(feet, t, fz, t_on, preload, ki)
%GS3DX_TRAIL_LOAD  Apply trail-foot preload law to foot references (#10979).
%
%   FEET = GS3DX_TRAIL_LOAD(FEET, T, FZ, T_ON, PRELOAD, KI) returns 6xN
%   BalanceFootRef-style foot references.  FEET is lead xyz (rows 1:3) and
%   trail xyz (rows 4:6), z up.  T and FZ are 1xN contact time and vertical
%   force on the trail foot.  Optional T_ON, PRELOAD and KI are forwarded to
%   GS3DX_TRAIL_PRELOAD.  Row 6 becomes the input row 6 minus preload height;
%   rows 1:5 are unchanged.
%
%   Error id gs3dx:preload.

    assert(size(feet, 1) == 6, 'gs3dx:preload', 'FEET must be 6xN');
    n = size(feet, 2);
    assert(numel(t) == n, 'gs3dx:preload', 'T must match FEET column count');
    assert(numel(fz) == n, 'gs3dx:preload', 'FZ must match FEET column count');

    switch nargin
        case 3
            h = gs3dx_trail_preload(t, fz);
        case 4
            h = gs3dx_trail_preload(t, fz, t_on);
        case 5
            h = gs3dx_trail_preload(t, fz, t_on, preload);
        otherwise
            h = gs3dx_trail_preload(t, fz, t_on, preload, ki);
    end
    feet = gs3dx_trail_foot_reference(feet, h);
end
