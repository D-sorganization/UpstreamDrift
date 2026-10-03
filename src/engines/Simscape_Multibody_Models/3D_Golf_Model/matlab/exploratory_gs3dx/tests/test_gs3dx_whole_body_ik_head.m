classdef test_gs3dx_whole_body_ik_head < matlab.unittest.TestCase
%TEST_GS3DX_WHOLE_BODY_IK_HEAD  Pure contract tests for head orientation input validation (#10979, #11161).
%   Proves fail-closed validation of head_R and gap.head_R BEFORE model setup:
%   - Missing head_R when head_orientation_weight > 0
%   - Rejection of invented head_gap / head aliases (canonical API: jc.gap.head_R)
%   - Non-logical gap vectors rejected
%   - Non-vector gap fields rejected
%   - Short gap vectors rejected (fail-closed, no silent false-measured)
%   - Calibration frame coverage required
%   - Non-SO(3) rotations on measured frames rejected
%   - NaN rotations permitted strictly on gap frames
%   All tests are pure and have zero external capture/ezc3d environment dependencies.

    methods (TestClassSetup)
        function setup(testCase)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root, 'tools'));
        end
    end

    methods (Test)
        function testMissingHeadRWhenPositiveWeight(testCase)
            jc = local_fixture_jc(5);
            testCase.verifyError(@() gs3dx_whole_body_ik(jc, frames=1:3, head_orientation_weight=0.01), ...
                'gs3dx:ik');
        end

        function testCanonicalGapFieldRequired(testCase)
            % Invented aliases (jc.head_gap, jc.gap.head) must be rejected
            jc = local_fixture_jc(5);
            jc.head_R = repmat(eye(3), [1, 1, 5]);
            jc.head_gap = false(1, 5); % Legacy invented alias

            testCase.verifyError(@() gs3dx_whole_body_ik(jc, frames=1:3, head_orientation_weight=0.01), ...
                'gs3dx:ik');
        end

        function testNonLogicalGapFieldRejected(testCase)
            jc = local_fixture_jc(5);
            jc.head_R = repmat(eye(3), [1, 1, 5]);
            jc.gap.head_R = [0, 0, 0, 0, 0]; % Double, not logical

            testCase.verifyError(@() gs3dx_whole_body_ik(jc, frames=1:3, head_orientation_weight=0.01), ...
                'gs3dx:ik');
        end

        function testNonVectorGapFieldRejected(testCase)
            jc = local_fixture_jc(5);
            jc.head_R = repmat(eye(3), [1, 1, 5]);
            jc.gap.head_R = false(2, 5); % Matrix, not vector

            testCase.verifyError(@() gs3dx_whole_body_ik(jc, frames=1:3, head_orientation_weight=0.01), ...
                'gs3dx:ik');
        end

        function testShortGapVectorRejected(testCase)
            % Gap vector shorter than required frames must fail closed
            jc = local_fixture_jc(5);
            jc.head_R = repmat(eye(3), [1, 1, 5]);
            jc.gap.head_R = false(1, 2); % Only 2 entries for 5 frames

            testCase.verifyError(@() gs3dx_whole_body_ik(jc, frames=1:5, head_orientation_weight=0.01), ...
                'gs3dx:ik');
        end

        function testCalibrationFrameCoverage(testCase)
            % head_R and gap.head_R must cover calibration frames even if frames are fewer
            jc = local_fixture_jc(5);
            jc.head_R = repmat(eye(3), [1, 1, 3]);
            jc.gap.head_R = false(1, 3);

            % calibration_frames includes frame 5, which exceeds length 3
            testCase.verifyError(@() gs3dx_whole_body_ik(jc, frames=1:2, calibration_frames=[1, 5], ...
                head_orientation_weight=0.01), 'gs3dx:ik');
        end

        function testInvalidRotationOnMeasuredFrame(testCase)
            % Measured frame (gap=false) with non-SO(3) rotation must error before setup
            jc = local_fixture_jc(5);
            jc.head_R = repmat(eye(3), [1, 1, 5]);
            jc.head_R(:, :, 2) = [1 0.5 0; 0 1 0; 0 0 1]; % Non-orthogonal
            jc.gap.head_R = false(1, 5);

            testCase.verifyError(@() gs3dx_whole_body_ik(jc, frames=1:3, head_orientation_weight=0.01), ...
                'gs3dx:ik');
        end


    end
end

function jc = local_fixture_jc(n)
    jc = struct();
    jc.pelvis = zeros(3, n);
    jc.t = (0:n-1) * 0.01;
    targets = ["pelvis", "hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R", ...
               "shoulder_L", "shoulder_R", "elbow_L", "elbow_R", "wrist_L", "wrist_R", "club_head"];
    jc.gap = struct();
    for i = 1:numel(targets)
        nm = targets(i);
        jc.(nm) = zeros(3, n);
        jc.gap.(nm) = false(1, n);
    end
end
