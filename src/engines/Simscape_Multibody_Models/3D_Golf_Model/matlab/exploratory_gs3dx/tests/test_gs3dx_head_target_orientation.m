classdef test_gs3dx_head_target_orientation < matlab.unittest.TestCase
%TEST_GS3DX_HEAD_TARGET_ORIENTATION  Unit tests for calibrated head target orientation (#10979, #11161).
%   Proves address frame identity, relative rotation compounding,
%   marker-triad permutation invariance, gap handling, and fail-closed
%   validation on non-SO(3) inputs.

    methods (TestClassSetup)
        function setup(testCase)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function testAddressFrameIdentity(testCase)
            % At address frame, target rotation must identically match model head rotation
            F_addr = local_euler(15, -20, 35);
            R_model_addr = local_euler(-10, 5, 25);
            F = cat(3, local_euler(0, 0, 0), F_addr, local_euler(30, 10, -5));

            [R_target, gaps, meta] = gs3dx_head_target_orientation(F, 2, R_model_addr);

            testCase.verifyFalse(gaps(2), 'Address frame must not be a gap');
            testCase.verifyEqual(R_target(:, :, 2), R_model_addr, 'AbsTol', 1e-12, ...
                'R_target at address frame must match R_model_head_address');
            testCase.verifyEqual(meta.F_address, F_addr, 'AbsTol', 1e-12);
            testCase.verifyEqual(meta.R_model_head_address, R_model_addr, 'AbsTol', 1e-12);
            testCase.verifyEqual(meta.address_frame, 2);
        end

        function testMeaningfulRotationsCompounding(testCase)
            % For any cluster rotation R_rel = F(f) * F_addr', target must be R_rel * R_model_addr
            F_addr = local_euler(10, 20, 30);
            R_model_addr = local_euler(-5, 15, -25);

            % Create known relative rotations
            R_rel1 = local_euler(25, 0, 0);   % pure yaw
            R_rel2 = local_euler(0, -35, 0);  % pure pitch
            R_rel3 = local_euler(0, 0, 45);   % pure roll
            R_rel4 = local_euler(15, -20, 30); % compound 3D

            F1 = R_rel1 * F_addr;
            F2 = R_rel2 * F_addr;
            F3 = R_rel3 * F_addr;
            F4 = R_rel4 * F_addr;
            F = cat(3, F_addr, F1, F2, F3, F4);

            [R_target, gaps] = gs3dx_head_target_orientation(F, 1, R_model_addr);

            testCase.verifyFalse(any(gaps), 'All frames should be valid');
            testCase.verifyEqual(R_target(:, :, 1), R_model_addr, 'AbsTol', 1e-12);
            testCase.verifyEqual(R_target(:, :, 2), R_rel1 * R_model_addr, 'AbsTol', 1e-12);
            testCase.verifyEqual(R_target(:, :, 3), R_rel2 * R_model_addr, 'AbsTol', 1e-12);
            testCase.verifyEqual(R_target(:, :, 4), R_rel3 * R_model_addr, 'AbsTol', 1e-12);
            testCase.verifyEqual(R_target(:, :, 5), R_rel4 * R_model_addr, 'AbsTol', 1e-12);

            % Verify every frame is strictly in SO(3)
            for k = 1:5
                Rk = R_target(:, :, k);
                testCase.verifyEqual(det(Rk), 1.0, 'AbsTol', 1e-12);
                testCase.verifyEqual(Rk' * Rk, eye(3), 'AbsTol', 1e-12);
            end
        end

        function testClusterTriadPermutationInvariance(testCase)
            % Permuting or rigidly rotating the cluster definition basis does not
            % change R_rel = (F * P) * (F_addr * P)' = F * P * P' * F_addr' = F * F_addr'
            F_addr = local_euler(12, -18, 42);
            R_model_addr = local_euler(5, -10, 15);
            R_rel = local_euler(-20, 30, -10);
            F_cur = R_rel * F_addr;

            F = cat(3, F_addr, F_cur);
            [R_target_base, ~] = gs3dx_head_target_orientation(F, 1, R_model_addr);

            % Right-multiply by arbitrary proper permutation/rotation matrix P in SO(3)
            P = [0 1 0; 0 0 1; 1 0 0]; % cyclic permutation
            F_perm = cat(3, F_addr * P, F_cur * P);
            [R_target_perm, ~] = gs3dx_head_target_orientation(F_perm, 1, R_model_addr);

            testCase.verifyEqual(R_target_perm, R_target_base, 'AbsTol', 1e-12, ...
                'Permuting cluster triad must yield identical target orientations');
        end

        function testGapMaskingAndMissingFrames(testCase)
            % Missing or nonfinite cluster frames produce NaN in R_target and gap=true
            F_addr = local_euler(10, 10, 10);
            R_model_addr = local_euler(5, 5, 5);
            F_valid = local_euler(20, 20, 20);

            F = cat(3, F_addr, nan(3, 3), F_valid, inf(3, 3));
            cluster_valid = [true, false, true, false];

            [R_target, gaps] = gs3dx_head_target_orientation(F, 1, R_model_addr, cluster_valid);

            testCase.verifyEqual(gaps, [false, true, false, true]);
            testCase.verifyTrue(all(isfinite(R_target(:, :, 1)), 'all'));
            testCase.verifyTrue(all(isnan(R_target(:, :, 2)), 'all'));
            testCase.verifyTrue(all(isfinite(R_target(:, :, 3)), 'all'));
            testCase.verifyTrue(all(isnan(R_target(:, :, 4)), 'all'));
        end

        function testFailClosedInvalidCalibration(testCase)
            % Invalid F_address or R_model_head_address throws fail-closed error
            F_addr_good = eye(3);
            R_model_good = eye(3);
            F = cat(3, eye(3), eye(3));

            % Out-of-bounds address frame index
            testCase.verifyError(@() gs3dx_head_target_orientation(F, 0, R_model_good), ...
                'gs3dx:head_orientation:invalid_address');
            testCase.verifyError(@() gs3dx_head_target_orientation(F, 3, R_model_good), ...
                'gs3dx:head_orientation:invalid_address');

            % Improper rotation det = -1 at address
            F_improper = eye(3);
            F_improper(1, 1) = -1;
            F_bad = cat(3, F_improper, eye(3));
            testCase.verifyError(@() gs3dx_head_target_orientation(F_bad, 1, R_model_good), ...
                'gs3dx:head_orientation:invalid_calibration');

            % Non-orthogonal R_model_head_address
            R_model_bad = [1 0.5 0; 0 1 0; 0 0 1];
            testCase.verifyError(@() gs3dx_head_target_orientation(F, 1, R_model_bad), ...
                'gs3dx:head_orientation:invalid_calibration');

            % Nonfinite R_model_head_address
            R_model_nan = eye(3);
            R_model_nan(2, 2) = NaN;
            testCase.verifyError(@() gs3dx_head_target_orientation(F, 1, R_model_nan), ...
                'gs3dx:head_orientation:invalid_calibration');
        end
    end
end

function R = local_euler(yaw_deg, pitch_deg, roll_deg)
    y = deg2rad(yaw_deg); p = deg2rad(pitch_deg); r = deg2rad(roll_deg);
    Rz = [cos(y) -sin(y) 0; sin(y) cos(y) 0; 0 0 1];
    Ry = [cos(p) 0 sin(p); 0 1 0; -sin(p) 0 cos(p)];
    Rx = [1 0 0; 0 cos(r) -sin(r); 0 sin(r) cos(r)];
    R = Rz * Ry * Rx;
end
