classdef test_gs3dx_head_input_data < matlab.unittest.TestCase
%TEST_GS3DX_HEAD_INPUT_DATA  Unit tests for head orientation input validation (#10979, #11161).
%   Proves independent known rotations, inactive baseline preservation, fail-closed
%   field and dimension validation, canonical gap formatting, frame bound checks,
%   proper SO(3) enforcement, and rejection of invalid weights or complex numbers.
%
%   Biomechanical Scope:
%     Cluster rotation matrices represent relative cluster alignment, NOT anatomical
%     skull qualification.

    methods (TestClassSetup)
        function setup(testCase)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function testKnownRealRotationsAndDirectData(testCase)
            % 3 frames with known orthonormal SO(3) rotations from basis oracle
            N = 3;
            pelvis = [0.1 0.2 0.3; 0.0 0.1 0.0; 0.9 0.9 0.9];
            R1 = eye(3);
            R2 = local_rot_oracle(3, pi / 2);  % 90 deg around Z
            R3 = local_rot_oracle(1, pi / 4);  % 45 deg around X
            R_all = cat(3, R1, R2, R3);
            gap.head_R = [false, false, false];
            jc = struct('pelvis', pelvis, 'head_R', R_all, 'gap', gap);

            data = gs3dx_head_input_data(jc, [1, 2, 3], [1], 1.5);
            testCase.verifyTrue(data.active);
            testCase.verifyEqual(data.R, double(R_all), 'AbsTol', 1e-12);
            testCase.verifyEqual(data.gaps, [false, false, false]);
            testCase.verifyTrue(islogical(data.gaps));
        end

        function testInactiveMinimalStruct(testCase)
            % Weight 0 and missing head_R returns active=false without evaluating jc
            jc_empty = struct();
            data1 = gs3dx_head_input_data(jc_empty, [1, 2], [], 0);
            testCase.verifyFalse(data1.active);
            testCase.verifyEmpty(data1.R);
            testCase.verifyEmpty(data1.gaps);

            jc_dummy = struct('unrelated', 42);
            data2 = gs3dx_head_input_data(jc_dummy, [1], [1], 0);
            testCase.verifyFalse(data2.active);
            testCase.verifyEmpty(data2.R);
            testCase.verifyEmpty(data2.gaps);
        end

        function testMissingFieldsActive(testCase)
            pelvis = zeros(3, 3);
            R = repmat(eye(3), 1, 1, 3);
            gap.head_R = false(1, 3);

            % Missing pelvis
            jc_no_pelvis = struct('head_R', R, 'gap', gap);
            testCase.verifyError(@() gs3dx_head_input_data(jc_no_pelvis, [1, 2], [1], 0), 'gs3dx:ik');

            % Missing head_R when weight > 0
            jc_no_head = struct('pelvis', pelvis, 'gap', gap);
            testCase.verifyError(@() gs3dx_head_input_data(jc_no_head, [1, 2], [1], 1.0), 'gs3dx:ik');

            % Missing gap struct or gap.head_R
            jc_no_gap = struct('pelvis', pelvis, 'head_R', R);
            testCase.verifyError(@() gs3dx_head_input_data(jc_no_gap, [1, 2], [1], 1.0), 'gs3dx:ik');

            % Non-struct jc
            testCase.verifyError(@() gs3dx_head_input_data(42, [1, 2], [1], 1.0), 'gs3dx:ik');
        end

        function testGapTypeAndShapeValidation(testCase)
            pelvis = zeros(3, 3);
            R = repmat(eye(3), 1, 1, 3);

            % Gap not logical
            gap_num.head_R = [0, 0, 0];
            testCase.verifyError(@() gs3dx_head_input_data(struct('pelvis', pelvis, 'head_R', R, 'gap', gap_num), [1, 2], [], 1.0), 'gs3dx:ik');

            % Gap not vector (2x2 matrix)
            gap_mat.head_R = false(2, 2);
            testCase.verifyError(@() gs3dx_head_input_data(struct('pelvis', pelvis, 'head_R', R, 'gap', gap_mat), [1, 2], [], 1.0), 'gs3dx:ik');

            % Gap wrong numel
            gap_short.head_R = false(1, 2);
            testCase.verifyError(@() gs3dx_head_input_data(struct('pelvis', pelvis, 'head_R', R, 'gap', gap_short), [1, 2], [], 1.0), 'gs3dx:ik');

            % Rejection of aliases (head_gap instead of canonical gap.head_R)
            jc_alias = struct('pelvis', pelvis, 'head_R', R, 'head_gap', false(1, 3));
            testCase.verifyError(@() gs3dx_head_input_data(jc_alias, [1, 2], [], 1.0), 'gs3dx:ik');
        end

        function testHeadRShortRelativePelvis(testCase)
            pelvis = zeros(3, 4);  % N = 4
            R_short = repmat(eye(3), 1, 1, 2);  % Only 2 frames
            gap.head_R = false(1, 4);

            % head_R short relative to pelvis N
            testCase.verifyError(@() gs3dx_head_input_data(struct('pelvis', pelvis, 'head_R', R_short, 'gap', gap), [1, 2], [1], 1.0), 'gs3dx:ik');

            % 4D head_R array rejected (ndims > 3)
            R_4d = repmat(eye(3), [1, 1, 4, 2]);
            testCase.verifyError(@() gs3dx_head_input_data(struct('pelvis', pelvis, 'head_R', R_4d, 'gap', gap), [1, 2], [1], 1.0), 'gs3dx:ik');
        end

        function testMeasuredSO3Failures(testCase)
            pelvis = zeros(3, 3);
            gap.head_R = [false, false, false];

            % Reflection matrix (det = -1)
            R_refl = cat(3, eye(3), diag([1, 1, -1]), eye(3));
            testCase.verifyError(@() gs3dx_head_input_data(struct('pelvis', pelvis, 'head_R', R_refl, 'gap', gap), [1, 2], [], 1.0), 'gs3dx:ik');

            % Non-orthogonal matrix (sheared)
            R_skew = cat(3, eye(3), [1 0.2 0; 0 1 0; 0 0 1], eye(3));
            testCase.verifyError(@() gs3dx_head_input_data(struct('pelvis', pelvis, 'head_R', R_skew, 'gap', gap), [1, 2], [], 1.0), 'gs3dx:ik');

            % NaN in measured frame (~gap)
            R_nan = cat(3, eye(3), nan(3, 3), eye(3));
            testCase.verifyError(@() gs3dx_head_input_data(struct('pelvis', pelvis, 'head_R', R_nan, 'gap', gap), [1, 2], [], 1.0), 'gs3dx:ik');
        end

        function testNaNOnlyGapsValid(testCase)
            % Frame 2 is marked gap and contains NaNs; frames 1 and 3 are valid
            pelvis = zeros(3, 3);
            R_gap_nan = cat(3, eye(3), nan(3, 3), local_rot_oracle(2, 0.3));
            gap.head_R = [false, true, false];

            data = gs3dx_head_input_data(struct('pelvis', pelvis, 'head_R', R_gap_nan, 'gap', gap), [1, 2, 3], [1], 1.0);
            testCase.verifyTrue(data.active);
            testCase.verifyEqual(data.gaps, [false, true, false]);
            testCase.verifyTrue(all(isnan(data.R(:, :, 2)), 'all'));
            testCase.verifyEqual(data.R(:, :, 1), eye(3), 'AbsTol', 1e-12);
        end

        function testNonvectorFrameInputsRejected(testCase)
            jc = struct('pelvis', zeros(3, 3), ...
                'head_R', repmat(eye(3), 1, 1, 3), ...
                'gap', struct('head_R', false(1, 3)));
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1 2; 2 3], [], 1), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1 2], [1 2; 2 3], 1), 'gs3dx:ik');
        end

        function testFrameBoundsAndCalibration(testCase)
            N = 3;
            pelvis = zeros(3, N);
            R = repmat(eye(3), 1, 1, N);
            gap.head_R = false(1, N);
            jc = struct('pelvis', pelvis, 'head_R', R, 'gap', gap);

            % Calibration frame 5 exceeds N=3
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 2], [5], 1.0), 'gs3dx:ik');

            % Calibration frame non-positive or non-integer
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 2], [0], 1.0), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 2], [1.5], 1.0), 'gs3dx:ik');

            % Requested frames beyond N or empty
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 4], [], 1.0), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_head_input_data(jc, [], [], 1.0), 'gs3dx:ik');

            % Empty calibration frames allowed
            data = gs3dx_head_input_data(jc, [1, 2], [], 1.0);
            testCase.verifyTrue(data.active);
        end

        function testComplexInputsAndNegativeOrNaNWeights(testCase)
            pelvis = zeros(3, 3);
            R = repmat(eye(3), 1, 1, 3);
            gap.head_R = false(1, 3);
            jc = struct('pelvis', pelvis, 'head_R', R, 'gap', gap);

            % Missing weight explicitly throws error
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 2], []), 'gs3dx:ik');

            % Negative, NaN, Inf, non-scalar, or complex weights fail
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 2], [], -0.5), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 2], [], NaN), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 2], [], Inf), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 2], [], [1, 2]), 'gs3dx:ik');
            testCase.verifyError(@() gs3dx_head_input_data(jc, [1, 2], [], 1 + 1i), 'gs3dx:ik');

            % Complex pelvis coordinates fail
            jc_c_pelvis = jc; jc_c_pelvis.pelvis(1, 1) = 1 + 1i;
            testCase.verifyError(@() gs3dx_head_input_data(jc_c_pelvis, [1, 2], [], 1.0), 'gs3dx:ik');

            % Complex head rotation coordinates fail
            jc_c_head = jc; jc_c_head.head_R(1, 1, 1) = 1 + 1i;
            testCase.verifyError(@() gs3dx_head_input_data(jc_c_head, [1, 2], [], 1.0), 'gs3dx:ik');
        end
    end
end

function R = local_rot_oracle(axis_idx, theta_rad)
    % Independent rotation matrix oracle for canonical coordinate axes
    c = cos(theta_rad); s = sin(theta_rad);
    switch axis_idx
        case 1
            R = [1 0 0; 0 c -s; 0 s c];
        case 2
            R = [c 0 s; 0 1 0; -s 0 c];
        case 3
            R = [c -s 0; s c 0; 0 0 1];
    end
end
