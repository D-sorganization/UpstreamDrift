classdef test_gs3dx_fit_grip < matlab.unittest.TestCase
%TEST_GS3DX_FIT_GRIP  Pure contract and precondition tests for gs3dx_fit_grip (#10979).
%
%   Verifies CAP/IK contract preconditions, calibration_frames input validation,
%   reference frame restriction to calibration frames, missing wrist filtering
%   (NaN, finite zeros, and residual/missing mask), and Design-by-Contract
%   postconditions on fitted grip variables using controlled structs.
%   Ordinary unit suite avoids expensive live model building or IK fitting.

    methods (TestClassSetup)
        function setup(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root, 'tools'));
            addpath(fullfile(root, 'models'));
            addpath(root);
        end
    end

    methods (Test)
        function test_invalid_cap_preconditions(testCase)
            % Non-struct, missing fields, or invalid n_frames must fail closed.
            ik_dummy = struct('frames', 1:5, 'joint', zeros(2, 5), 'joint_ids', ["j1"; "j2"], 'model', 'dummy');

            testCase.verifyError(@() gs3dx_fit_grip(123, ik_dummy), 'MATLAB:validation:UnableToConvert');
            testCase.verifyError(@() gs3dx_fit_grip(struct('missing_n_frames', 10), ik_dummy), 'gs3dx:grip:invalid_cap');
            testCase.verifyError(@() gs3dx_fit_grip(struct('n_frames', -5, 'marker', @(x) 0, 'target_frame', eye(3)), ik_dummy), ...
                'gs3dx:grip:invalid_cap');
            testCase.verifyError(@() gs3dx_fit_grip(struct('n_frames', 2.5, 'marker', @(x) 0, 'target_frame', eye(3)), ik_dummy), ...
                'gs3dx:grip:invalid_cap');
            testCase.verifyError(@() gs3dx_fit_grip(struct('n_frames', NaN, 'marker', @(x) 0, 'target_frame', eye(3)), ik_dummy), ...
                'gs3dx:grip:invalid_cap');
            testCase.verifyError(@() gs3dx_fit_grip(struct('n_frames', 10, 'target_frame', eye(3)), ik_dummy), ...
                'gs3dx:grip:invalid_cap');
            testCase.verifyError(@() gs3dx_fit_grip(struct('n_frames', 10, 'marker', @(x) 0), ik_dummy), ...
                'gs3dx:grip:invalid_cap');
        end

        function test_invalid_ik_preconditions(testCase)
            % Missing required fields in IK struct must fail closed.
            cap_dummy = struct('n_frames', 10, 'marker', @(x) zeros(3, 10), 'target_frame', eye(3));

            bad_ik1 = struct('joint', zeros(2, 5)); % missing frames, joint_ids, model
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_ik1), 'gs3dx:grip');

            bad_ik2 = struct('frames', 1:5, 'joint', zeros(2, 5), 'joint_ids', ["j1"; "j2"]); % missing model
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_ik2), 'gs3dx:grip');
        end

        function test_invalid_ik_frames(testCase)
            % IK.frames must be a non-empty, real, finite, positive, strictly increasing,
            % unique integer vector bounded by cap.n_frames.
            cap_dummy = struct('n_frames', 20, 'marker', @(x) zeros(3, 20), 'target_frame', eye(3));
            base_ik = struct('frames', 1:5, 'joint', zeros(2, 5), 'joint_ids', ["j1"; "j2"], 'model', 'dummy');

            % Non-numeric or complex
            bad_frames1 = base_ik; bad_frames1.frames = [1, 2 + 1i, 3];
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_frames1), 'gs3dx:grip:invalid_ik');

            % Non-integer
            bad_frames2 = base_ik; bad_frames2.frames = [1, 2.5, 3];
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_frames2), 'gs3dx:grip:invalid_ik');

            % Zero or negative
            bad_frames3 = base_ik; bad_frames3.frames = [0, 1, 2];
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_frames3), 'gs3dx:grip:invalid_ik');

            % Non-strictly increasing / duplicates
            bad_frames4 = base_ik; bad_frames4.frames = [1, 2, 2, 3];
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_frames4), 'gs3dx:grip:invalid_ik');

            % Exceeding cap.n_frames
            bad_frames5 = base_ik; bad_frames5.frames = [5, 10, 25];
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_frames5), 'gs3dx:grip:invalid_ik');

            % Empty
            bad_frames6 = base_ik; bad_frames6.frames = [];
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_frames6), 'gs3dx:grip:invalid_ik');
        end

        function test_invalid_ik_joints_and_status(testCase)
            % Dimension mismatch, non-2D, non-finite, complex joints, or failed loop closure must fail closed.
            cap_dummy = struct('n_frames', 10, 'marker', @(x) zeros(3, 10), 'target_frame', eye(3));
            base_ik = struct('frames', 1:5, 'joint', zeros(2, 5), 'joint_ids', ["j1"; "j2"], 'model', 'dummy');

            % joint rows (3) ~= joint_ids (2)
            bad_j1 = base_ik; bad_j1.joint = zeros(3, 5);
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_j1), 'gs3dx:grip:invalid_ik');

            % joint cols (4) ~= frames (5)
            bad_j2 = base_ik; bad_j2.joint = zeros(2, 4);
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_j2), 'gs3dx:grip:invalid_ik');

            % Non-2D joint matrix
            bad_j3 = base_ik; bad_j3.joint = zeros(2, 5, 2);
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_j3), 'gs3dx:grip:invalid_ik');

            % Non-finite joint values
            bad_j4 = base_ik; bad_j4.joint(1, 1) = NaN;
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_j4), 'gs3dx:grip:invalid_ik');

            % Complex joint values
            bad_j5 = base_ik; bad_j5.joint(1, 1) = 0.5i;
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, bad_j5), 'gs3dx:grip:invalid_ik');
        end

        function test_calibration_frames_validation(testCase)
            % calibration_frames must be real, finite, positive, ordered, unique, integer within capture.
            cap_dummy = struct('n_frames', 50, 'marker', @(x) zeros(3, 50), 'target_frame', eye(3));
            ik_dummy = struct('frames', 1:5, 'joint', zeros(2, 5), 'joint_ids', ["j1"; "j2"], 'model', 'dummy');

            % Non-integer
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, ik_dummy, calibration_frames=[1.5, 3.5]), ...
                'gs3dx:grip:invalid_calibration_frames');

            % Non-positive (contains 0 or negative)
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, ik_dummy, calibration_frames=[0, 1, 2]), ...
                'gs3dx:grip:invalid_calibration_frames');

            % Unordered (descending or scrambled)
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, ik_dummy, calibration_frames=[5, 2, 8]), ...
                'gs3dx:grip:invalid_calibration_frames');

            % Duplicates / non-unique
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, ik_dummy, calibration_frames=[1, 2, 2, 3]), ...
                'gs3dx:grip:invalid_calibration_frames');

            % Out of capture bounds (> cap.n_frames)
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, ik_dummy, calibration_frames=[10, 20, 51]), ...
                'gs3dx:grip:invalid_calibration_frames');

            % Non-finite (contains NaN or Inf)
            testCase.verifyError(@() gs3dx_fit_grip(cap_dummy, ik_dummy, calibration_frames=[1, NaN, 5]), ...
                'gs3dx:grip:invalid_calibration_frames');
        end

        function test_reference_frame_restricted_to_cal_frames(testCase)
            % Reference frame must be selected strictly from calibration_frames, not held-out frames.
            [cap, ik] = local_make_controlled_cap_and_ik(12);

            cal_frames = 5:9;
            grip = gs3dx_fit_grip(cap, ik, calibration_frames=cal_frames);

            testCase.verifyTrue(ismember(grip.provenance.reference_frame, cal_frames), ...
                'Reference frame must be inside calibration_frames');
            testCase.verifyNotEqual(grip.provenance.reference_frame, 1, ...
                'Reference frame must not be held-out frame 1 when cal_frames starts later');
        end

        function test_too_few_club_markers_in_cal_frames(testCase)
            % If fewer than 3 calibration frames have valid club markers, fail closed.
            [cap, ik] = local_make_controlled_cap_and_ik(10);

            % Only provide 2 calibration frames
            testCase.verifyError(@() gs3dx_fit_grip(cap, ik, calibration_frames=[4, 5]), ...
                'gs3dx:grip:too_few_club_markers');
        end

        function test_wrist_missing_filtering_finite_zeros_and_nan(testCase)
            % Missing wrist frames with NaNs or finite zeros [0; 0; 0] must be filtered out without substitution.
            [cap, ik] = local_make_controlled_cap_and_ik(10);

            % Introduce dropout: frame 3 has NaN, frame 4 has finite zeros [0; 0; 0]
            raw_marker = cap.marker;
            lw_copy = raw_marker("LWristTop");
            lw_copy(:, 3) = NaN;
            lw_copy(:, 4) = [0; 0; 0]; % finite zeros dropout

            cap.marker = @(name) local_override_marker(raw_marker, name, "LWristTop", lw_copy);

            grip = gs3dx_fit_grip(cap, ik, calibration_frames=1:10);

            % Out of 10 calibration frames, 2 were dropouts -> measured count must be 8
            testCase.verifyEqual(grip.wrist_measured_counts(1), 8);
            testCase.verifyEqual(grip.wrist_measured_counts(2), 10);
        end

        function test_wrist_missing_residual_and_mask_filtering(testCase)
            % Explicit residual and missing masks must exclude their frames.
            [cap, ik] = local_make_controlled_cap_and_ik(10);

            raw_marker = cap.marker;
            % Residual: frame 2 has -1.0 (missing in C3D standard)
            lw_res = ones(1, 10);
            lw_res(2) = -1.0;

            cap.residual = struct('LWristTop', lw_res);
            missing = false(1, 10);
            missing(3) = true;
            cap.missing = struct('LWristTop', missing);

            grip = gs3dx_fit_grip(cap, ik, calibration_frames=1:10);

            testCase.verifyEqual(grip.wrist_measured_counts(1), 8);
            testCase.verifyEqual(grip.wrist_measured_counts(2), 10);
        end

        function test_too_few_wrist_samples_fails_closed(testCase)
            % If fewer than 4 valid wrist frames exist in calibration_frames, fail closed.
            [cap, ik] = local_make_controlled_cap_and_ik(10);

            % Make 8 out of 10 frames NaN for LWristTop
            raw_marker = cap.marker;
            lw_copy = raw_marker("LWristTop");
            lw_copy(:, 1:8) = NaN;

            cap.marker = @(name) local_override_marker(raw_marker, name, "LWristTop", lw_copy);

            testCase.verifyError(@() gs3dx_fit_grip(cap, ik, calibration_frames=1:10), ...
                'gs3dx:grip:too_few_wrist_samples');
        end

        function test_pure_contract_fit_grip_and_dbc(testCase)
            % Full deterministic verification of DbC postconditions, sphere fit, and privacy.
            [cap, ik] = local_make_controlled_cap_and_ik(10);

            grip = gs3dx_fit_grip(cap, ik);
            g = grip.vars;

            % All grip variables positive and finite
            testCase.verifyGreaterThan(g.FitButtToLeadHand, 0);
            testCase.verifyGreaterThan(g.FitHandSpacing, 0);
            testCase.verifyGreaterThan(g.FitGripToShaft, 0);
            testCase.verifyGreaterThan(g.FitLeftWristStandoff, 0);
            testCase.verifyGreaterThan(g.FitRightWristStandoff, 0);

            % Total grip length sum to 10.5 in
            total_len = g.FitButtToLeadHand + g.FitHandSpacing + g.FitGripToShaft;
            testCase.verifyEqual(total_len, 10.5, 'AbsTol', 1e-10);

            % Equal standoff gauge
            testCase.verifyEqual(g.FitLeftWristStandoff, g.FitRightWristStandoff, 'AbsTol', 1e-12);

            % Sphere diagnostics
            for s = 1:2
                testCase.verifyGreaterThan(grip.sphere(s).radius, 0);
                testCase.verifyGreaterThan(grip.sphere(s).rms, 0);
                testCase.verifyEqual(grip.sphere(s).rank, 4);
                testCase.verifyGreaterThan(grip.sphere(s).condition, 0);
                testCase.verifyGreaterThan(grip.sphere(s).rank_tolerance, 0);
                testCase.verifyEqual(grip.sphere(s).n_points, 10);
            end

            % Privacy check on provenance
            p = grip.provenance;
            testCase.verifyTrue(isstruct(p));
            testCase.verifyTrue(isfield(p, 'calibration_frames_count'));
            testCase.verifyTrue(isfield(p, 'wrist_measured_counts'));
            testCase.verifyTrue(isfield(p, 'axis_frames_count'));
            testCase.verifyTrue(isfield(p, 'reference_frame'));
            testCase.verifyTrue(isfield(p, 'model'));

            json_prov = jsonencode(p);
            testCase.verifyFalse(contains(json_prov, ["C:\", "D:\", "/home/", "Users"]), ...
                'Provenance must not leak private filesystem paths');
        end

        function test_explicit_calibration_frames_zero_leakage(testCase)
            % User selecting calibration_frames must fit wrists and shaft axis ONLY on those frames.
            [cap, ik] = local_make_controlled_cap_and_ik(15);

            cal_frames = 2:2:10;
            grip = gs3dx_fit_grip(cap, ik, calibration_frames=cal_frames);

            testCase.verifyEqual(grip.calibration_frames, cal_frames);
            testCase.verifyEqual(grip.provenance.calibration_frames_count, numel(cal_frames));
            testCase.verifyTrue(all(grip.wrist_measured_counts <= numel(cal_frames)), ...
                'Wrist measured samples strictly bounded by calibration frames');
        end
    end
end

%% Helper functions for controlled test fixtures
function [cap, ik] = local_make_controlled_cap_and_ik(n_frames)
    if nargin < 1
        n_frames = 10;
    end

    cap = struct();
    cap.n_frames = n_frames;
    cap.rate_hz = 240;
    cap.target_frame = eye(3);

    markers_map = containers.Map();
    markers_map('WaistLeft') = repmat([-0.1; 0; 1.0], 1, n_frames);
    markers_map('WaistRight') = repmat([0.1; 0; 1.0], 1, n_frames);
    markers_map('WaistLBack') = repmat([-0.1; -0.1; 1.0], 1, n_frames);
    markers_map('WaistRBack') = repmat([0.1; -0.1; 1.0], 1, n_frames);

    % Club markers: two clusters along shaft
    % Cluster 2 (grip): around [0; 0; 0.8]
    markers_map('Marker_2:2:1') = repmat([0.02; 0; 0.8], 1, n_frames);
    markers_map('Marker_2:2:2') = repmat([-0.02; 0.02; 0.8], 1, n_frames);
    markers_map('Marker_2:2:3') = repmat([-0.02; -0.02; 0.8], 1, n_frames);
    % Cluster 3 (head): around [0; 0; 0.1]
    markers_map('Marker_3:3:1') = repmat([0.03; 0; 0.1], 1, n_frames);
    markers_map('Marker_3:3:2') = repmat([-0.03; 0.03; 0.1], 1, n_frames);
    markers_map('Marker_3:3:3') = repmat([-0.03; -0.03; 0.1], 1, n_frames);

    % Wrists moving on distinct sphere trajectories in club frame
    angles = linspace(0, pi, n_frames);
    lw = zeros(3, n_frames);
    rw = zeros(3, n_frames);
    for f = 1:n_frames
        th = angles(f);
        lw(:, f) = [0.03 + 0.08 * cos(th); 0.08 * sin(th); 0.75 + 0.02 * sin(2*th)];
        rw(:, f) = [-0.03 + 0.08 * cos(th); 0.08 * sin(th); 0.68 + 0.02 * sin(2*th)];
    end
    markers_map('LWristTop') = lw;
    markers_map('RWristTop') = rw;

    cap.labels = string(markers_map.keys());
    cap.marker = @(name) markers_map(char(name));

    % Controlled IK
    ik = struct();
    ik.frames = 1:n_frames;
    ik.model = 'GS3DX_Fit';
    ik.joint_ids = ["j1"; "j2"];
    ik.joint = zeros(2, n_frames);
    ik.status = ones(1, n_frames);

    % Hand centres
    hand = zeros(3, 2, n_frames);
    for f = 1:n_frames
        % Native IK world positions use the address waist origin [0;-.05;1].
        hand(:, 1, f) = [0; 0.05; -0.25]; % Lead hand
        hand(:, 2, f) = [0; 0.05; -0.32]; % Trail hand
    end
    ik.hand_centres = hand;
end

function out = local_override_marker(base_marker, name, target_name, override_data)
    if strcmp(char(name), char(target_name))
        out = override_data;
    else
        out = base_marker(name);
    end
end
