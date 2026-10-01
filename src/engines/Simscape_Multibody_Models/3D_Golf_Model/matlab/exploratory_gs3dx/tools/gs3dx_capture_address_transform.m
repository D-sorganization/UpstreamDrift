function tf = gs3dx_capture_address_transform(cap)
%GS3DX_CAPTURE_ADDRESS_TRANSFORM  Transform capture points to address target frame (#10979).
%
%   TF = GS3DX_CAPTURE_ADDRESS_TRANSFORM(CAP) derives the transform logic
%   into the address target frame [facing, toward target, up] with the origin
%   at the address waist centre (mean of the four waist markers at frame 1).
%   Shared by GS3DX_CAPTURE_JOINT_CENTRES and GS3DX_CAPTURE_HEAD_FRAME.
%
%   TF fields:
%     .origin   3x1 address waist centre in capture axes
%     .S        3x3 basis matrix [facing, toward target, up]
%     .local    function handle @(p) S.' * (p - origin)
%     .fill     function handle @(x) fillmissing(x, 'linear', 2, 'EndValues', 'nearest')
%     .raw      function handle @(name) fill(cap.marker(name))
%     .m        function handle @(name) local(raw(name))
%
%   See also GS3DX_CAPTURE_JOINT_CENTRES, GS3DX_CAPTURE_HEAD_FRAME.

    arguments
        cap (1,1) struct
    end
    assert(isfield(cap, 'target_frame'), 'gs3dx:transform', 'cap missing target_frame');
    assert(isfield(cap, 'marker'), 'gs3dx:transform', 'cap missing marker accessor');

    S = cap.target_frame;
    fill = @(x) fillmissing(x, 'linear', 2, 'EndValues', 'nearest');
    raw = @(name) fill(cap.marker(name));
    waist = (raw("WaistLeft") + raw("WaistRight") + raw("WaistLBack") + raw("WaistRBack")) / 4;
    origin = waist(:, 1);
    local = @(p) S.' * (p - origin);
    m = @(name) local(raw(name));

    tf = struct('origin', origin, 'S', S, 'local', local, 'fill', fill, 'raw', raw, 'm', m);
end
