function [mass, position] = gs3dx_capture_segments(opts)
%GS3DX_CAPTURE_SEGMENTS  De Leva segment COM positions from the capture (#11011).
%
%   [MASS, POSITION] = GS3DX_CAPTURE_SEGMENTS() returns point-mass segment
%   masses (kg, Px1) and positions (3xPxN, m) in the capture lab frame,
%   before the address target-frame rotation.  Segment layout matches
%   GS3DX_KINEMATIC_GRF (head, trunk, L/R limb chain, club head and grip).
%
%   Options match GS3DX_KINEMATIC_GRF: file, body_mass, hip_drop,
%   club_head_mass, club_rest_mass, trunk ("c7" or "joint_centres").
%
%   Error id gs3dx:segments.

    arguments
        opts.file (1,:) char = ''
        opts.body_mass (1,1) double {mustBePositive} = 80
        opts.hip_drop (1,1) double {mustBeNonnegative} = 0.08
        opts.club_head_mass (1,1) double {mustBeNonnegative} = 0.25
        opts.club_rest_mass (1,1) double {mustBeNonnegative} = 0.137
        opts.trunk (1,1) string {mustBeMember(opts.trunk, ["c7" "joint_centres"])} = "c7"
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
    if opts.trunk == "c7"
        trunk = along(m("BackTop"), waist - drop, c.trunk);
    else
        jc = gs3dx_capture_joint_centres(cap);
        back = @(p) cap.target_frame * p + waist(:, 1);
        trunk = along(back((jc.shoulder_L + jc.shoulder_R) / 2), back((jc.hip_L + jc.hip_R) / 2), c.trunk);
    end
    parts = {
        f.head,  (m("HeadTop") + m("HeadFront") + m("HeadSide")) / 3
        f.trunk, trunk
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
    assert(abs(sum(w) - 1) < 1e-9, 'gs3dx:segments', 'Segment fractions sum to %.6f, not 1', sum(w));
    M = opts.body_mass + opts.club_head_mass + opts.club_rest_mass;
    parts = [parts; {opts.club_head_mass / opts.body_mass, fill(cap.club_head); ...
                     opts.club_rest_mass / opts.body_mass, fill(cap.club_grip)}];
    w = cell2mat(parts(:, 1)) * opts.body_mass / M;

    P = size(parts, 1);
    n = cap.n_frames;
    position = zeros(3, P, n);
    for i = 1:P
        position(:, i, :) = parts{i, 2};
    end
    mass = w * M;
    assert(numel(mass) == P && all(mass > 0), 'gs3dx:segments', 'Segment masses must be positive');
    assert(all(isfinite(position(:))) && all(isfinite(mass)), 'gs3dx:segments', ...
        'Segment masses and positions must be finite');
end
