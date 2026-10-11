function tests = test_gs3dx_foot_marker_frame
%TEST_GS3DX_FOOT_MARKER_FRAME  Unit tests for pure foot marker frame helper (#10979, #11161).
%   Proves elevated ankle calibration, 3D rotation recovery including roll,
%   proper SO(3) mean, fail-closed measurement validation, and gap masking.
    tests = functiontests(localfunctions);
end

function setupOnce(t)
    t.TestData.original_path = path;
    tools_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools');
    addpath(tools_dir);
end

function teardownOnce(t)
    path(t.TestData.original_path);
end

%% Helper: Synthetic foot marker triad generator
function [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, origin, ankle_height, toe_spread, foot_len)
    if nargin < 3, ankle_height = 0.07; end
    if nargin < 4, toe_spread = 0.08; end
    if nargin < 5, foot_len = 0.20; end

    % In shoe frame: sole is in x-y plane (+z up), +x forward, +y left
    % Ankle marker sits at [-foot_len * 0.3; 0; ankle_height] (elevated!)
    % ToeIn sits at [+foot_len * 0.7; +toe_spread/2; 0]
    % ToeOut sits at [+foot_len * 0.7; -toe_spread/2; 0]
    p_ankle_local = [-foot_len * 0.3; 0; ankle_height];
    p_toein_local = [foot_len * 0.7; toe_spread / 2; 0];
    p_toeout_local = [foot_len * 0.7; -toe_spread / 2; 0];

    N = size(R_world, 3);
    ankle = zeros(3, N);
    toe_in = zeros(3, N);
    toe_out = zeros(3, N);

    for f = 1:N
        R = R_world(:, :, f);
        p0 = origin(:, f);
        ankle(:, f) = p0 + R * p_ankle_local;
        toe_in(:, f) = p0 + R * p_toein_local;
        toe_out(:, f) = p0 + R * p_toeout_local;
    end
end

%% 1. Elevated Ankle with Flat Address Yields Horizontal Forward and Vertical Up Sole
function testElevatedAnkleFlatAddressCalibration(t)
    % A flat shoe with sole in X-Y plane (+z up, +x forward along +X)
    R_flat = eye(3);
    origin = [0; 0; 0];
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_flat, origin, 0.07, 0.08, 0.20);

    % Address frame is 1, address yaw is 0 deg
    [R, gaps_out, meta] = gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false, ...
        'address_frames', 1, 'address_yaw_deg', 0);

    verifyFalse(t, gaps_out(1));
    % At address, R must be strictly identity:
    % forward axis (+x) is horizontal +X = [1; 0; 0]
    % sole normal (+z) is vertical +Z = [0; 0; 1]
    verifyEqual(t, R(:, :, 1), eye(3), 'AbsTol', 1e-12);
    verifyEqual(t, meta.F_address' * meta.F_address, eye(3), 'AbsTol', 1e-12);
    verifyEqual(t, det(meta.F_address), 1.0, 'AbsTol', 1e-12);
end

%% 2. Known Rigid Triad 3D Rotation Recovery (Including Roll)
function testKnownRigidTriadRotationRecovery(t)
    % Test poses across 4 frames:
    % Frame 1: Flat Address (yaw = 10 deg)
    % Frame 2: Pure Roll (30 deg about x)
    % Frame 3: Pure Pitch (25 deg about y)
    % Frame 4: Combined 3D Roll/Pitch/Yaw
    yaw1 = 10;
    R1 = [cosd(yaw1) -sind(yaw1) 0; sind(yaw1) cosd(yaw1) 0; 0 0 1];

    roll2 = 30;
    R2 = R1 * [1 0 0; 0 cosd(roll2) -sind(roll2); 0 sind(roll2) cosd(roll2)];

    pitch3 = 25;
    R3 = R1 * [cosd(pitch3) 0 sind(pitch3); 0 1 0; -sind(pitch3) 0 cosd(pitch3)];

    R4 = R1 * [cosd(15) 0 sind(15); 0 1 0; -sind(15) 0 cosd(15)] * ...
              [1 0 0; 0 cosd(-20) -sind(-20); 0 sind(-20) cosd(-20)];

    R_known = cat(3, R1, R2, R3, R4);
    origin = zeros(3, 4);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_known, origin, 0.065, 0.08, 0.22);

    [R_est, gaps_out, ~] = gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 4), ...
        'address_frames', 1, 'address_yaw_deg', yaw1);

    verifyFalse(t, any(gaps_out));
    for f = 1:4
        verifyEqual(t, R_est(:, :, f), R_known(:, :, f), 'AbsTol', 1e-11);
        % Must be proper SO(3)
        verifyEqual(t, det(R_est(:, :, f)), 1.0, 'AbsTol', 1e-12);
        verifyEqual(t, norm(R_est(:, :, f)' * R_est(:, :, f) - eye(3), 'fro'), 0.0, 'AbsTol', 1e-12);
    end
end

%% 3. Address Yaw in Horizontal Plane Application
function testAddressYawApplication(t)
    R_flat = eye(3);
    origin = [0; 0; 0];
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_flat, origin);

    % Explicit address yaw of 45 deg
    yaw_deg = 45;
    [R, ~, meta] = gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false, ...
        'address_frames', 1, 'address_yaw_deg', yaw_deg);

    verifyEqual(t, meta.address_yaw_deg, 45);
    % +x must point along [cosd(45); sind(45); 0]
    expected_x = [cosd(45); sind(45); 0];
    verifyEqual(t, R(:, 1, 1), expected_x, 'AbsTol', 1e-12);
    % +z must point along vertical [0; 0; 1]
    verifyEqual(t, R(:, 3, 1), [0; 0; 1], 'AbsTol', 1e-12);
end

%% 4. Proper SO(3) Mean Over Multiple Address Frames
function testProperMeanMultipleAddressFrames(t)
    % 3 address frames with slight perturbations
    R1 = eye(3);
    R2 = [cosd(2) -sind(2) 0; sind(2) cosd(2) 0; 0 0 1];
    R3 = [cosd(-1) -sind(-1) 0; sind(-1) cosd(-1) 0; 0 0 1];
    R_world = cat(3, R1, R2, R3);
    origin = zeros(3, 3);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, origin);

    [R, ~, meta] = gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', [1, 2, 3], 'address_yaw_deg', 0);

    verifyEqual(t, det(meta.F_address), 1.0, 'AbsTol', 1e-12);
    verifyEqual(t, norm(meta.F_address' * meta.F_address - eye(3), 'fro'), 0.0, 'AbsTol', 1e-12);
    for f = 1:3
        verifyEqual(t, det(R(:, :, f)), 1.0, 'AbsTol', 1e-12);
    end
end

%% 5. Explicit Gaps Permit Missing Measurements Without Inventing Data
function testExplicitGapsPermitMissingFrames(t)
    R_world = repmat(eye(3), [1 1 3]);
    origin = zeros(3, 3);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, origin);

    % Frame 2 has NaN coordinates and explicit gap=true
    ankle(:, 2) = NaN;
    toe_in(:, 2) = NaN;
    toe_out(:, 2) = NaN;
    gaps = [false, true, false];

    [R, gaps_out, ~] = gs3dx_foot_marker_frame(ankle, toe_in, toe_out, gaps, ...
        'address_frames', 1, 'address_yaw_deg', 0);

    verifyFalse(t, gaps_out(1));
    verifyTrue(t, gaps_out(2));
    verifyFalse(t, gaps_out(3));

    verifyTrue(t, all(isnan(R(:, :, 2)), 'all'));
    verifyEqual(t, R(:, :, 1), eye(3), 'AbsTol', 1e-12);
    verifyEqual(t, R(:, :, 3), eye(3), 'AbsTol', 1e-12);
end

%% 6. Non-Finite Measurement Without Gap Throws Error
function testNonFiniteMeasurementWithoutGapThrows(t)
    R_world = repmat(eye(3), [1 1 2]);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, zeros(3, 2));

    % Frame 2 has NaN coordinates but gap=false -> MUST fail closed
    ankle(1, 2) = NaN;
    gaps = [false, false];

    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, gaps), ...
        'gs3dx:foot_marker_frame:invalid_measurement');
end

%% 7. Degenerate Ankle-Toe Distance Throws Error
function testDegenerateAnkleToeDistanceThrows(t)
    R_world = eye(3);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, zeros(3, 1));
    % Place ankle exactly at toe mean
    toe_mean = (toe_in + toe_out) / 2;
    ankle = toe_mean;

    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false), ...
        'gs3dx:foot_marker_frame:degenerate_triad');
end

%% 8. Degenerate ToeIn-ToeOut Spread Throws Error
function testDegenerateToeSpreadThrows(t)
    R_world = eye(3);
    [ankle, toe_in, ~] = local_build_foot_markers(R_world, zeros(3, 1));
    % Make ToeOut identical to ToeIn
    toe_out = toe_in;

    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false), ...
        'gs3dx:foot_marker_frame:degenerate_triad');
end

%% 9. Collinear Foot Markers Throws Error
function testCollinearFootMarkersThrows(t)
    % ToeIn, ToeOut, and Ankle all lie on the same line
    ankle = [0; 0; 0];
    toe_in = [0.1; 0; 0];
    toe_out = [0.2; 0; 0];

    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false), ...
        'gs3dx:foot_marker_frame:degenerate_triad');
end

%% 10. Invalid Address Frames Throws Error
function testInvalidAddressFramesThrows(t)
    R_world = repmat(eye(3), [1 1 2]);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, zeros(3, 2));

    % Out of bounds address frame
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, [false, false], 'address_frames', 99), ...
        'gs3dx:foot_marker_frame:invalid_address');

    % All address frames gap-filled
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, [true, false], 'address_frames', 1), ...
        'gs3dx:foot_marker_frame:no_valid_address');
end

%% 11. Dimension Mismatch Throws Error
function testDimensionMismatchThrows(t)
    ankle = zeros(3, 5);
    toe_in = zeros(3, 4); % mismatch
    toe_out = zeros(3, 5);

    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out), ...
        'gs3dx:foot_marker_frame:invalid_dimension');
end

%% 12. Malformed Yaw Argument Throws Error
function testMalformedYawThrows(t)
    R_world = repmat(eye(3), [1 1 2]);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, zeros(3, 2));

    % Non-scalar vector yaw
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, [false, false], ...
        'address_yaw_deg', [10, 20]), 'gs3dx:foot_marker_frame:invalid_yaw');

    % Non-finite NaN yaw
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, [false, false], ...
        'address_yaw_deg', NaN), 'gs3dx:foot_marker_frame:invalid_yaw');

    % Non-finite Inf yaw
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, [false, false], ...
        'address_yaw_deg', Inf), 'gs3dx:foot_marker_frame:invalid_yaw');

    % Complex yaw
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, [false, false], ...
        'address_yaw_deg', 10 + 2i), 'gs3dx:foot_marker_frame:invalid_yaw');

    % Empty yaw succeeds (falls back to horizontal vector calculation)
    [R, ~, meta] = gs3dx_foot_marker_frame(ankle, toe_in, toe_out, [false, false], ...
        'address_yaw_deg', []);
    verifyTrue(t, isfield(meta, 'address_yaw_deg'));
    verifyTrue(t, isfinite(meta.address_yaw_deg));
    verifyEqual(t, size(R), [3, 3, 2]);
end

%% 13. Malformed Address Frames Throws Error
function testMalformedAddressFramesThrows(t)
    R_world = repmat(eye(3), [1 1 3]);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, zeros(3, 3));

    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', [1 2; 2 3]), 'gs3dx:foot_marker_frame:invalid_address');

    % Empty address frames
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', []), 'gs3dx:foot_marker_frame:invalid_address');

    % Non-integer address frame
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', 1.5), 'gs3dx:foot_marker_frame:invalid_address');

    % Non-positive address frame (zero or negative)
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', 0), 'gs3dx:foot_marker_frame:invalid_address');
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', -1), 'gs3dx:foot_marker_frame:invalid_address');

    % Out of bounds address frame (> N)
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', 4), 'gs3dx:foot_marker_frame:invalid_address');

    % Non-finite address frame (NaN or Inf)
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', NaN), 'gs3dx:foot_marker_frame:invalid_address');
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', Inf), 'gs3dx:foot_marker_frame:invalid_address');

    % Complex address frame
    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, false(1, 3), ...
        'address_frames', 1 + 1i), 'gs3dx:foot_marker_frame:invalid_address');
end

%% 14. Degenerate Address Mean Rejects Non-Unique Rank-Deficient Orientation
function testDegenerateAddressMeanThrows(t)
    % Two address frames with opposing 180 deg roll
    % Frame 1: Identity
    % Frame 2: 180 deg roll (about x-axis: [1 0 0; 0 -1 0; 0 0 -1])
    % Their sum has rank 1 (singular values [2, 0, 0]), which lacks a unique SO(3) mean
    R1 = eye(3);
    R2 = [1 0 0; 0 -1 0; 0 0 -1];
    R_world = cat(3, R1, R2);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, zeros(3, 2));

    verifyError(t, @() gs3dx_foot_marker_frame(ankle, toe_in, toe_out, [false, false], ...
        'address_frames', [1, 2], 'address_yaw_deg', 0), ...
        'gs3dx:foot_marker_frame:degenerate_mean');
end

%% 15. Explicit Gap Complex Markers Never Generate Complex Rotations
function testExplicitGapComplexMarkersNeverGenerateComplexRotations(t)
    R_world = repmat(eye(3), [1 1 3]);
    [ankle, toe_in, toe_out] = local_build_foot_markers(R_world, zeros(3, 3));

    % Frame 2 is an explicit gap with complex marker coordinates
    ankle(:, 2) = ankle(:, 2) + 0.05i;
    toe_in(:, 2) = toe_in(:, 2) - 0.02i;
    toe_out(:, 2) = toe_out(:, 2) + 0.03i;
    gaps = [false, true, false];

    [R, gaps_out, ~] = gs3dx_foot_marker_frame(ankle, toe_in, toe_out, gaps, ...
        'address_frames', 1, 'address_yaw_deg', 0);

    verifyFalse(t, gaps_out(1));
    verifyTrue(t, gaps_out(2));
    verifyFalse(t, gaps_out(3));

    % Must never generate complex rotations in R
    verifyTrue(t, isreal(R));
    verifyTrue(t, all(isnan(R(:, :, 2)), 'all'));
    verifyEqual(t, R(:, :, 1), eye(3), 'AbsTol', 1e-12);
    verifyEqual(t, R(:, :, 3), eye(3), 'AbsTol', 1e-12);
end

%% 16. Missing Frame 1 Foot Markers Fails Closed in Capture Joint Centres
function testFrame1MissingFailsClosed(t)
    cap_missing = local_build_synthetic_cap(25, true);
    verifyError(t, @() gs3dx_capture_joint_centres(cap_missing), ...
        'gs3dx:capture_joint_centres:frame1_missing');

    % When frame 1 is measured, calibration succeeds and stores per-foot metadata
    cap_valid = local_build_synthetic_cap(25, false);
    jc = gs3dx_capture_joint_centres(cap_valid);
    verifyTrue(t, isfield(jc, 'foot_calibration'));
    verifyTrue(t, isfield(jc.foot_calibration, 'L'));
    verifyTrue(t, isfield(jc.foot_calibration, 'R'));
    verifyTrue(t, isfield(jc.foot_calibration.L, 'assumption'));
    verifyTrue(t, isfield(jc.foot_calibration.R, 'assumption'));
    verifyTrue(t, contains(jc.foot_calibration.L.assumption, 'Flat sole at address is an ASSUMPTION'));
    verifyTrue(t, contains(jc.foot_calibration.R.assumption, 'Flat sole at address is an ASSUMPTION'));
    verifyEqual(t, size(jc.foot_R_L), [3, 3, 25]);
    verifyEqual(t, size(jc.foot_R_R), [3, 3, 25]);
end

%% Helper: Synthetic capture generator for capture joint centres tests
function cap = local_build_synthetic_cap(n_frames, gap_frame1_foot)
    if nargin < 1, n_frames = 25; end
    if nargin < 2, gap_frame1_foot = false; end

    cap.n_frames = n_frames;
    cap.rate_hz = 120;
    cap.impact_frame = 5;
    cap.target_frame = eye(3);
    cap.club_grip = zeros(3, n_frames);
    cap.club_head = zeros(3, n_frames);

    marker_map = containers.Map();
    names = [ ...
        "WaistLeft", "WaistRight", "WaistLBack", "WaistRBack", ...
        "BackTop", "BackLeft", "BackRight", ...
        "LShoulderTop", "RShoulderTop", "RShoulderBack", ...
        "LKneeOut", "RKneeOut", "LAnkleOut", "RAnkleOut", ...
        "LToeIn", "LToeOut", "RToeIn", "RToeOut", ...
        "LElbowOut", "RElbowOut", "LWristTop", "RWristTop"];

    for i = 1:numel(names)
        nm = names(i);
        marker_map(char(nm)) = repmat([0; 0; 0], 1, n_frames);
    end

    marker_map('WaistLeft')   = repmat([0;  0.15; 0.9], 1, n_frames);
    marker_map('WaistRight')  = repmat([0; -0.15; 0.9], 1, n_frames);
    marker_map('WaistLBack')  = repmat([-0.1;  0.15; 0.9], 1, n_frames);
    marker_map('WaistRBack')  = repmat([-0.1; -0.15; 0.9], 1, n_frames);
    marker_map('BackTop')     = repmat([-0.05; 0; 1.4], 1, n_frames);
    marker_map('BackLeft')    = repmat([-0.08;  0.12; 1.3], 1, n_frames);
    marker_map('BackRight')   = repmat([-0.08; -0.12; 1.3], 1, n_frames);
    marker_map('LShoulderTop')= repmat([0;  0.2; 1.4], 1, n_frames);
    marker_map('RShoulderTop')= repmat([0; -0.2; 1.4], 1, n_frames);
    marker_map('RShoulderBack')= repmat([-0.05; -0.2; 1.38], 1, n_frames);
    marker_map('LKneeOut')    = repmat([0;  0.15; 0.5], 1, n_frames);
    marker_map('RKneeOut')    = repmat([0; -0.15; 0.5], 1, n_frames);
    marker_map('LAnkleOut')   = repmat([-0.05;  0.15; 0.08], 1, n_frames);
    marker_map('RAnkleOut')   = repmat([-0.05; -0.15; 0.08], 1, n_frames);
    marker_map('LToeIn')      = repmat([0.15;  0.18; 0.01], 1, n_frames);
    marker_map('LToeOut')     = repmat([0.15;  0.10; 0.01], 1, n_frames);
    marker_map('RToeIn')      = repmat([0.15; -0.10; 0.01], 1, n_frames);
    marker_map('RToeOut')     = repmat([0.15; -0.18; 0.01], 1, n_frames);
    marker_map('LElbowOut')   = repmat([0;  0.25; 1.1], 1, n_frames);
    marker_map('RElbowOut')   = repmat([0; -0.25; 1.1], 1, n_frames);
    marker_map('LWristTop')   = repmat([0.1;  0.15; 0.8], 1, n_frames);
    marker_map('RWristTop')   = repmat([0.1; -0.15; 0.8], 1, n_frames);

    if gap_frame1_foot
        m_ltoein = marker_map('LToeIn');
        m_ltoein(:, 1) = NaN;
        marker_map('LToeIn') = m_ltoein;
    end

    cap.marker = @(name) marker_map(char(name));
end
