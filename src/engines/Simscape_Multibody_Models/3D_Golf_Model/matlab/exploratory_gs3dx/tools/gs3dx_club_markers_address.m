function [X, ok] = gs3dx_club_markers_address(cap)
%GS3DX_CLUB_MARKERS_ADDRESS  Six club markers in the address target frame (#11160).
%
%   [X, OK] = GS3DX_CLUB_MARKERS_ADDRESS(CAP) maps GS3DX_CLUB_MARKERS into
%   the capture address target frame (the IK world) via
%   GS3DX_CAPTURE_ADDRESS_TRANSFORM.

    arguments
        cap (1,1) struct
    end
    tf = gs3dx_capture_address_transform(cap);
    [X, ok] = gs3dx_club_markers(cap);
    local = tf.local;
    for f = find(ok)
        for k = 1:6
            X(:, k, f) = local(X(:, k, f));
        end
    end
end
