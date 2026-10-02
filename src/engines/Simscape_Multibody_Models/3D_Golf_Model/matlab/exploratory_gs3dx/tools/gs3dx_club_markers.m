function [X, ok] = gs3dx_club_markers(cap)
%GS3DX_CLUB_MARKERS  Six club-marker tracks in capture lab coordinates (#11160).
%
%   [X, OK] = GS3DX_CLUB_MARKERS(CAP) returns X as 3 x 6 x n_frames and OK
%   as 1 x n_frames, true only when all six markers are finite (no gap-fill).

    arguments
        cap (1,1) struct
    end
    idx = find(startsWith(cap.labels, ["Marker_2:2:", "Marker_3:3:"]));
    assert(numel(idx) == 6, 'gs3dx:club', 'Expected six club markers, found %d', numel(idx));
    X = zeros(3, 6, cap.n_frames);
    for k = 1:6
        X(:, k, :) = reshape(cap.marker(cap.labels(idx(k))), 3, 1, []);
    end
    ok = reshape(all(isfinite(X), [1 2]), 1, []);
end
