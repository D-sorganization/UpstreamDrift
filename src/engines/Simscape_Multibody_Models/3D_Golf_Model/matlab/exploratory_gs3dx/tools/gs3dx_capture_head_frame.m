function hf = gs3dx_capture_head_frame(cap)
%GS3DX_CAPTURE_HEAD_FRAME  Head orientation and position from capture markers (#10979).
%
%   HF = GS3DX_CAPTURE_HEAD_FRAME(CAP) constructs a per-frame head reference
%   frame from the three head markers HeadTop, HeadFront, and HeadSide of CAP
%   (GS3DX_CAPTURE_MARKERS).  Positions and orientations are expressed in the
%   address target frame [facing, toward target, up] with the origin at the
%   address waist centre (matching GS3DX_CAPTURE_JOINT_CENTRES).  Gaps are
%   filled linearly.
%
%   Orientation construction:
%     The right-handed orthonormal basis [forward, left, up] is constructed
%     via Gram-Schmidt orthogonalization from marker differences:
%       1. left: from the sagittal midline (midpoint of HeadTop and
%          HeadFront) to HeadSide: unit(HeadSide - (HeadTop + HeadFront)/2).
%       2. up: from HeadFront to HeadTop, made orthogonal to left:
%          unit(up_raw - (up_raw . left) * left).
%       3. forward: cross(left, up) pointing forward out of the face.
%     At address, forward (.R(:,1,1)) has a positive component along +X,
%     left (.R(:,2,1)) along +Y, and up (.R(:,3,1)) along +Z.
%
%   HF fields:
%     .centre        3 x n, mean of the three head markers in the address
%                    target frame (m)
%     .R             3 x 3 x n, right-handed orthonormal rotation matrix
%                    columns [forward, left, up] in the address target frame
%     .R_rel         3 x 3 x n, rotation from address: R(:,:,t) * R(:,:,1)'
%     .gap           1 x n logical, true where any head marker was gap-filled
%     .t             1 x n time vector (s) from CAP
%     .impact_frame  impact frame index (1-based) from CAP
%
%   See also GS3DX_CAPTURE_MARKERS, GS3DX_CAPTURE_JOINT_CENTRES,
%            GS3DX_CAPTURE_ADDRESS_TRANSFORM.

    arguments
        cap (1,1) struct
    end

    head_markers = ["HeadTop", "HeadFront", "HeadSide"];
    assert(isfield(cap, 'labels'), 'gs3dx:head', 'cap must have .labels field');
    for name = head_markers
        assert(any(cap.labels == name), 'gs3dx:head', 'Head marker %s not found in cap', name);
    end
    assert(isfield(cap, 'marker'), 'gs3dx:head', 'cap must have .marker accessor');
    assert(isfield(cap, 'n_frames'), 'gs3dx:head', 'cap must have .n_frames');
    assert(isfield(cap, 'impact_frame'), 'gs3dx:head', 'cap must have .impact_frame');

    raw_top = cap.marker("HeadTop");
    raw_front = cap.marker("HeadFront");
    raw_side = cap.marker("HeadSide");

    assert(~all(isnan(raw_top(:))), 'gs3dx:head', 'HeadTop marker has no valid samples');
    assert(~all(isnan(raw_front(:))), 'gs3dx:head', 'HeadFront marker has no valid samples');
    assert(~all(isnan(raw_side(:))), 'gs3dx:head', 'HeadSide marker has no valid samples');

    n = cap.n_frames;
    gap_top = any(isnan(raw_top), 1);
    gap_front = any(isnan(raw_front), 1);
    gap_side = any(isnan(raw_side), 1);
    hf.gap = gap_top | gap_front | gap_side;

    tf = gs3dx_capture_address_transform(cap);
    ht = tf.local(tf.fill(raw_top));
    hf_pt = tf.local(tf.fill(raw_front));
    hs = tf.local(tf.fill(raw_side));

    hf.centre = (ht + hf_pt + hs) / 3;

    left_raw = hs - (ht + hf_pt) / 2;
    left = local_unit(left_raw);
    up_raw = ht - hf_pt;
    up = local_unit(up_raw - sum(up_raw .* left, 1) .* left);
    fwd = cross(left, up, 1);

    hf.R = permute(cat(3, fwd, left, up), [1 3 2]);

    R1 = hf.R(:, :, 1);
    hf.R_rel = pagemtimes(hf.R, permute(R1, [2 1 3]));

    if isfield(cap, 't')
        hf.t = cap.t;
    else
        assert(isfield(cap, 'rate_hz'), 'gs3dx:head', 'cap must have .rate_hz');
        hf.t = (0:n - 1) / cap.rate_hz;
    end
    hf.impact_frame = cap.impact_frame;

    assert(all(isfinite(hf.centre(:))), 'gs3dx:head', 'Postcondition: head centre has non-finite samples');
    assert(all(isfinite(hf.R(:))), 'gs3dx:head', 'Postcondition: head rotation matrix has non-finite samples');
    assert(hf.R(1, 1, 1) > 0, 'gs3dx:head', 'Postcondition: forward axis does not point along +X at address');
    assert(hf.R(3, 3, 1) > 0, 'gs3dx:head', 'Postcondition: up axis does not point along +Z at address');
end

function u = local_unit(v)
    u = v ./ vecnorm(v);
end
