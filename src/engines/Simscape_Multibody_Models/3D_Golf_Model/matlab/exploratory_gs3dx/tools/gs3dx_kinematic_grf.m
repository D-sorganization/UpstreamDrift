function k = gs3dx_kinematic_grf(opts)
%GS3DX_KINEMATIC_GRF  Total ground reaction force from the capture, without force plates (#11011).
%
%   K = GS3DX_KINEMATIC_GRF() estimates the whole-body centre of mass (COM)
%   of the tour-average driver capture from its markers and the de Leva
%   segment table (GS3DX_ANTHROPOMETRY), and from Newton's second law the
%   total external force on the golfer:
%
%       GRF(t) = M * (a_com(t) - g),   g = [0 0 -9.81] m/s^2 (Z-up).
%
%   The system is the body and the club it holds; only the feet touch the
%   ground, so this is the sum of both feet's ground reaction forces,
%   except for the ball's impulse on the club at impact (~3 N*s, which the
%   filter spreads to ~0.05 BW around impact).  The split between the feet
%   and the centre of pressure are not observable this way; the contact
%   model predicts those.
%
%   Segments (proximal -> distal marker proxies, COM at the de Leva
%   fraction .com from the proximal end):
%     head       mean of HeadTop/HeadFront/HeadSide (the COM itself)
%     trunk      BackTop (C7) -> waist centre lowered by hip_drop
%     upper arm  LShoulderTop / RShoulderBack -> ElbowOut
%                (RShoulderTop is missing in 80% of frames)
%     forearm    ElbowOut -> WristTop
%     hand       WristTop (the COM itself; 0.6% of body mass)
%     thigh      waist side marker lowered by hip_drop -> KneeOut
%     shank      KneeOut -> AnkleOut
%     foot       AnkleOut -> mean(ToeIn, ToeOut)
%     club       club_head_mass at the head marker cluster, club_rest_mass
%                (shaft, grip) at the grip cluster; the head cluster has a
%                33-frame gap in the follow-through (filled linearly)
%   Markers sit on the skin, not on joint centres, so each segment COM is
%   a proxy: the GRF shape and timing are robust, a few percent of body
%   weight in the level are not (docs/ANTHROPOMETRY.md).
%
%   Gaps are filled linearly, the COM is low-pass filtered (zero-lag 4th
%   order Butterworth at cutoff_hz) and differentiated twice.
%
%   Options: file ('' = data/C3D_TA_Driver.c3d), body_mass (80 kg),
%   cutoff_hz (8), hip_drop (0.08 m, waist markers above the hip centres),
%   club_head_mass (0.25 kg) and club_rest_mass (0.137 kg: the GS3DX
%   model's shaft, grip parts and hand standoffs).
%
%   K fields: .t (s, 0 at the first frame), .com (3xN, m), .grf (3xN, N)
%   and .grf_bw (3xN, in body weights), all in the address target frame
%   [facing, toward target, up] (GS3DX_CAPTURE_MARKERS .target_frame; the
%   COM relative to the capture origin; body weight = body + club), .mass
%   (body + club, kg), .rate_hz,
%   .impact_frame, .address (mean vertical GRF, BW, over the first 0.1 s),
%   .peak (max vertical GRF up to impact, BW, and its time relative to
%   impact, s; the follow-through is left out because the club-head
%   cluster has a gap there).

    arguments
        opts.file (1,:) char = ''
        opts.body_mass (1,1) double {mustBePositive} = 80
        opts.cutoff_hz (1,1) double {mustBePositive} = 8
        opts.hip_drop (1,1) double {mustBeNonnegative} = 0.08
        opts.club_head_mass (1,1) double {mustBeNonnegative} = 0.25
        opts.club_rest_mass (1,1) double {mustBeNonnegative} = 0.137
    end
    cap = gs3dx_capture_markers(opts.file);
    a = gs3dx_anthropometry(opts.body_mass);
    f = a.fraction;
    c = a.com;
    fill = @(x) fillmissing(x, 'linear', 2, 'EndValues', 'nearest');
    m = @(name) fill(cap.marker(name));
    along = @(p, d, r) p + r * (d - p);
    drop = [0; 0; opts.hip_drop];

    waist = (m("WaistLeft") + m("WaistRight") + m("WaistLBack") + m("WaistRBack")) / 4;
    parts = {
        f.head,  (m("HeadTop") + m("HeadFront") + m("HeadSide")) / 3
        f.trunk, along(m("BackTop"), waist - drop, c.trunk)
        };
    shoulder = struct('L', "LShoulderTop", 'R', "RShoulderBack");
    hip = struct('L', "WaistLeft", 'R', "WaistRight");
    for s = ["L", "R"]
        elbow = m(s + "ElbowOut");
        wrist = m(s + "WristTop");
        knee = m(s + "KneeOut");
        ankle = m(s + "AnkleOut");
        toe = (m(s + "ToeIn") + m(s + "ToeOut")) / 2;
        parts = [parts; {
            f.upper_arm, along(m(shoulder.(s)), elbow, c.upper_arm)
            f.forearm,   along(elbow, wrist, c.forearm)
            f.hand,      wrist
            f.thigh,     along(m(hip.(s)) - drop, knee, c.thigh)
            f.shank,     along(knee, ankle, c.shank)
            f.foot,      along(ankle, toe, c.foot)}]; %#ok<AGROW>
    end
    w = cell2mat(parts(:, 1));
    assert(abs(sum(w) - 1) < 1e-9, 'gs3dx:kgrf', 'Segment fractions sum to %.6f, not 1', sum(w));
    M = opts.body_mass + opts.club_head_mass + opts.club_rest_mass;
    parts = [parts; {opts.club_head_mass / opts.body_mass, fill(cap.club_head); ...
                     opts.club_rest_mass / opts.body_mass, fill(cap.club_grip)}];
    w = cell2mat(parts(:, 1)) * opts.body_mass / M;
    com = zeros(3, cap.n_frames);
    for i = 1:numel(w)
        com = com + w(i) * parts{i, 2};
    end

    fs = cap.rate_hz;
    [b, aa] = butter(4, opts.cutoff_hz / (fs / 2));
    com_f = filtfilt(b, aa, com.').';
    dt = 1 / fs;
    acc = gradient(gradient(com_f, dt), dt);
    g = [0; 0; -9.81];
    k.t = (0:cap.n_frames - 1) * dt;
    S = cap.target_frame;
    k.com = S.' * com_f;
    k.grf = S.' * (M * (acc - g));
    k.grf_bw = k.grf / (M * norm(g));
    k.mass = M;
    k.rate_hz = fs;
    k.impact_frame = cap.impact_frame;
    k.address = mean(k.grf_bw(3, k.t <= 0.1));
    [pk, i] = max(k.grf_bw(3, 1:cap.impact_frame));
    k.peak = struct('vertical_bw', pk, 'time_to_impact', k.t(i) - k.t(cap.impact_frame));
    assert(all(isfinite(k.grf(:))), 'gs3dx:kgrf', 'Postcondition: GRF has non-finite samples');
end
