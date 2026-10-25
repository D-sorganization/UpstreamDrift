function q = gs3dx_capture_momentum(opts)
%GS3DX_CAPTURE_MOMENTUM  Capture angular momentum in the address frame (#11173).
%
%   Q = GS3DX_CAPTURE_MOMENTUM() estimates whole-body angular momentum from
%   de Leva point-mass segment COMs (GS3DX_CAPTURE_SEGMENTS).  Segment spin
%   is not included.  Each segment trajectory is low-pass filtered with the
%   same zero-lag 4th-order Butterworth as GS3DX_KINEMATIC_GRF, then
%   GS3DX_POINT_MOMENTUM forms H and HDOT.  Both are rotated into the address
%   target frame with S.' (same as the kinematic COM).
%
%   Options: file, body_mass, cutoff_hz (8), hip_drop, club_head_mass,
%   club_rest_mass, trunk ("c7" or "joint_centres", default "c7").
%
%   Q fields: .t (s), .h and .hdot (3xN, kg*m^2/s and kg*m^2/s^2).

    arguments
        opts.file (1,:) char = ''
        opts.body_mass (1,1) double {mustBePositive} = 80
        opts.cutoff_hz (1,1) double {mustBePositive} = 8
        opts.hip_drop (1,1) double {mustBeNonnegative} = 0.08
        opts.club_head_mass (1,1) double {mustBeNonnegative} = 0.25
        opts.club_rest_mass (1,1) double {mustBeNonnegative} = 0.137
        opts.trunk (1,1) string {mustBeMember(opts.trunk, ["c7" "joint_centres"])} = "c7"
    end

    cap = gs3dx_capture_markers(opts.file);
    [mass, pos] = gs3dx_capture_segments( ...
        file=opts.file, body_mass=opts.body_mass, hip_drop=opts.hip_drop, ...
        club_head_mass=opts.club_head_mass, club_rest_mass=opts.club_rest_mass, ...
        trunk=opts.trunk);

    fs = cap.rate_hz;
    [b, aa] = butter(4, opts.cutoff_hz / (fs / 2));
    P = size(pos, 2);
    n = size(pos, 3);
    posF = zeros(size(pos));
    for i = 1:P
        ri = reshape(pos(:, i, :), 3, n);
        posF(:, i, :) = filtfilt(b, aa, ri.').';
    end

    dt = 1 / fs;
    [h, hdot] = gs3dx_point_momentum(mass, posF, dt);

    S = cap.target_frame;
    q.t = (0:n - 1) * dt;
    q.h = S.' * h;
    q.hdot = S.' * hdot;
    assert(all(isfinite(q.h(:))) && all(isfinite(q.hdot(:))), 'gs3dx:momentum', ...
        'Postcondition: H and HDOT must be finite');
end
