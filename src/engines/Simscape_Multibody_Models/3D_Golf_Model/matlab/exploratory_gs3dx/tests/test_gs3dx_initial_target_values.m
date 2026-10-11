classdef test_gs3dx_initial_target_values < matlab.unittest.TestCase
%TEST_GS3DX_INITIAL_TARGET_VALUES  TDD test suite for gs3dx_initial_target_values (#10979).
%
%   Tests pure mapping of 48-variable solved joint kinematics to native
%   GS3DX_Human initial workspace and literal block target parameters.
%   Validates permutation invariance, whitelist enforcement, Rodrigues/Euler
%   conversions, gimbal singularity detection, and closed-arm preservation.

    properties (Constant)
        CANONICAL_KEYS = [
            "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Px.p"
            "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Py.p"
            "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Pz.p"
            "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_x"
            "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_y"
            "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_z"
            "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.q"
            "Hips and Torso Inputs/Neck Joint|Rx.q"
            "Hips and Torso Inputs/Neck Joint|Ry.q"
            "Hips and Torso Inputs/Spine Tilt Kinetically Driven/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
            "Hips and Torso Inputs/Spine Tilt Kinetically Driven/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
            "Hips and Torso Inputs/Torso Kinetically Driven/Revolute Joint/Kinetically Driven Revolute|Rz.q"
            "Left Elbow Joint/Revolute Joint/Kinetically Driven Revolute|Rz.q"
            "Left Forearm/Revolute Joint/Kinetically Driven Revolute|Rz.q"
            "Left Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
            "Left Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
            "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_x"
            "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_y"
            "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_z"
            "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.q"
            "Left Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
            "Left Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
            "Lower Body/L Midfoot Joint|Rz.q"
            "Lower Body/Left Ankle Joint/Kinetically Driven Universal Joint|Rx.q"
            "Lower Body/Left Ankle Joint/Kinetically Driven Universal Joint|Ry.q"
            "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_x"
            "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_y"
            "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_z"
            "Lower Body/Left Hip Joint/Kinetically Driven|S.q"
            "Lower Body/Left Knee Joint/Kinetically Driven Revolute|Rz.q"
            "Lower Body/R Midfoot Joint|Rz.q"
            "Lower Body/Right Ankle Joint/Kinetically Driven Universal Joint|Rx.q"
            "Lower Body/Right Ankle Joint/Kinetically Driven Universal Joint|Ry.q"
            "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_x"
            "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_y"
            "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_z"
            "Lower Body/Right Hip Joint/Kinetically Driven|S.q"
            "Lower Body/Right Knee Joint/Kinetically Driven Revolute|Rz.q"
            "Right Elbow Joint/Revolute Joint/Kinetically Driven Revolute|Rz.q"
            "Right Forearm/Revolute Joint/Kinetically Driven Revolute|Rz.q"
            "Right Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
            "Right Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
            "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_x"
            "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_y"
            "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_z"
            "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.q"
            "Right Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Rx.q"
            "Right Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Ry.q"
        ];
    end

    methods (TestClassSetup)
        function setupClass(~)
            exploratoryDir = fileparts(fileparts(mfilename('fullpath')));
            addpath(exploratoryDir);
            addpath(fullfile(exploratoryDir, 'tools'));
            gs3dx_setup();
        end
    end

    methods (Test)
        function testIdentityPoseQuietDiagnostic(testCase)
            [keys, units, q] = local_canonical_inputs();
            targets = gs3dx_initial_target_values(keys, units, q);

            testCase.verifyEqual(targets.qualification, 'PURE_MAPPING_ONLY');
            ws = targets.workspace_values;

            % Pelvis position [m] and zero velocity
            testCase.verifyEqual(ws.TranslationStartPositionX, 0.1, 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.TranslationStartPositionY, -0.2, 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.TranslationStartPositionZ, 0.85, 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.TranslationStartVelocityX, 0.0);
            testCase.verifyEqual(ws.TranslationStartVelocityY, 0.0);
            testCase.verifyEqual(ws.TranslationStartVelocityZ, 0.0);

            % Spherical identity angles map to 0 deg
            testCase.verifyEqual(ws.HipStartPositionX, 0.0, 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.HipStartPositionY, 0.0, 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.HipStartPositionZ, 0.0, 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.HipStartVelocityX, 0.0);
            testCase.verifyEqual(ws.HipStartVelocityY, 0.0);
            testCase.verifyEqual(ws.HipStartVelocityZ, 0.0);

            % Hips 3x1 and Ankles 2x1 shapes
            testCase.verifySize(ws.LHipStartPosition, [3, 1]);
            testCase.verifySize(ws.LHipStartVelocity, [3, 1]);
            testCase.verifyEqual(ws.LHipStartPosition, zeros(3, 1), 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.LHipStartVelocity, zeros(3, 1));

            testCase.verifySize(ws.RHipStartPosition, [3, 1]);
            testCase.verifySize(ws.RHipStartVelocity, [3, 1]);
            testCase.verifyEqual(ws.RHipStartPosition, zeros(3, 1), 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.RHipStartVelocity, zeros(3, 1));

            testCase.verifySize(ws.LAnkleStartPosition, [2, 1]);
            testCase.verifySize(ws.LAnkleStartVelocity, [2, 1]);
            testCase.verifyEqual(ws.LAnkleStartPosition, zeros(2, 1), 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.LAnkleStartVelocity, zeros(2, 1));

            testCase.verifySize(ws.RAnkleStartPosition, [2, 1]);
            testCase.verifySize(ws.RAnkleStartVelocity, [2, 1]);
            testCase.verifyEqual(ws.RAnkleStartPosition, zeros(2, 1), 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.RAnkleStartVelocity, zeros(2, 1));

            % Knee scalars
            testCase.verifyTrue(isscalar(ws.LKneeStartPosition));
            testCase.verifyTrue(isscalar(ws.LKneeStartVelocity));
            testCase.verifyEqual(ws.LKneeStartPosition, 0.0);
            testCase.verifyEqual(ws.LKneeStartVelocity, 0.0);
        end

        function testKnown90DegZRotation(testCase)
            [keys, units, q] = local_canonical_inputs();
            % Set Pelvis Spherical rotation to 90 deg around +Z
            q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_x") = 0.0;
            q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_y") = 0.0;
            q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_z") = 1.0;
            q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.q") = 90.0;

            targets = gs3dx_initial_target_values(keys, units, q);
            ws = targets.workspace_values;

            testCase.verifyEqual(ws.HipStartPositionX, 0.0, 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.HipStartPositionY, 0.0, 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.HipStartPositionZ, 90.0, 'AbsTol', 1e-10);
        end

        function testKnown45DegXRotation(testCase)
            [keys, units, q] = local_canonical_inputs();
            % Set Left Shoulder Spherical rotation to 45 deg around +X
            q(keys == "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_x") = 1.0;
            q(keys == "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_y") = 0.0;
            q(keys == "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_z") = 0.0;
            q(keys == "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.q") = 45.0;

            targets = gs3dx_initial_target_values(keys, units, q);
            ws = targets.workspace_values;

            testCase.verifyEqual(ws.LSStartPositionX, 45.0, 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.LSStartPositionY, 0.0, 'AbsTol', 1e-10);
            testCase.verifyEqual(ws.LSStartPositionZ, 0.0, 'AbsTol', 1e-10);
        end

        function testIndependentRotationOracleRoundtrip(testCase)
            [keys, units, q] = local_canonical_inputs();
            % Target intrinsic follower XYZ angles
            euler_target = [25.0; -35.0; 40.0];
            [ax, ang_deg] = local_euler_to_axis_angle_oracle(euler_target(1), euler_target(2), euler_target(3));

            q(keys == "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_x") = ax(1);
            q(keys == "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_y") = ax(2);
            q(keys == "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_z") = ax(3);
            q(keys == "Lower Body/Left Hip Joint/Kinetically Driven|S.q") = ang_deg;

            targets = gs3dx_initial_target_values(keys, units, q);
            testCase.verifyEqual(targets.workspace_values.LHipStartPosition, euler_target, 'AbsTol', 1e-8);
        end

        function testClosedRightArmPreservation(testCase)
            [keys, units, q] = local_canonical_inputs();
            % Assign distinct solved non-zero values to all closed right arm coordinates
            q(keys == "Right Elbow Joint/Revolute Joint/Kinetically Driven Revolute|Rz.q") = 31.5;
            q(keys == "Right Forearm/Revolute Joint/Kinetically Driven Revolute|Rz.q") = -18.2;
            q(keys == "Right Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Rx.q") = 7.4;
            q(keys == "Right Scapula Joint/Universal Joint/Kinetically Driven Universal Joint|Ry.q") = -4.3;
            q(keys == "Right Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Rx.q") = 12.8;
            q(keys == "Right Wrist and Hand/Universal Joint/Kinetically Driven Universal Joint|Ry.q") = -9.6;

            % Right Shoulder: 20 deg around [0, 0, 1]
            q(keys == "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_x") = 0.0;
            q(keys == "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_y") = 0.0;
            q(keys == "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_z") = 1.0;
            q(keys == "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.q") = 20.0;

            targets = gs3dx_initial_target_values(keys, units, q);
            ws = targets.workspace_values;

            % Verify none of the right arm coordinates are zeroed
            testCase.verifyEqual(ws.REStartPosition, 31.5, 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.RFStartPosition, -18.2, 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.RScapStartPositionX, 7.4, 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.RScapStartPositionY, -4.3, 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.RWStartPositionX, 12.8, 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.RWStartPositionY, -9.6, 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.RSStartPositionZ, 20.0, 'AbsTol', 1e-10);
        end

        function testLiteralTargetNeckAndMidfootValues(testCase)
            [keys, units, q] = local_canonical_inputs();
            q(keys == "Hips and Torso Inputs/Neck Joint|Rx.q") = 14.2;
            q(keys == "Hips and Torso Inputs/Neck Joint|Ry.q") = -8.7;
            q(keys == "Lower Body/L Midfoot Joint|Rz.q") = 5.3;
            q(keys == "Lower Body/R Midfoot Joint|Rz.q") = 3.9;

            targets = gs3dx_initial_target_values(keys, units, q);
            bp = targets.block_position_values;

            testCase.verifyEqual(numel(bp), 4);
            testCase.verifyTrue(all(ismember({'block', 'primitive', 'value', 'unit'}, fieldnames(bp))));

            neckRx = bp(string({bp.block}) == "Hips and Torso Inputs/Neck Joint" & string({bp.primitive}) == "Rx.q");
            neckRy = bp(string({bp.block}) == "Hips and Torso Inputs/Neck Joint" & string({bp.primitive}) == "Ry.q");
            lMid = bp(string({bp.block}) == "Lower Body/L Midfoot Joint" & string({bp.primitive}) == "Rz.q");
            rMid = bp(string({bp.block}) == "Lower Body/R Midfoot Joint" & string({bp.primitive}) == "Rz.q");

            testCase.verifyEqual(neckRx.value, 14.2, 'AbsTol', 1e-12);
            testCase.verifyEqual(neckRx.unit, 'deg');
            testCase.verifyEqual(neckRy.value, -8.7, 'AbsTol', 1e-12);
            testCase.verifyEqual(neckRy.unit, 'deg');
            testCase.verifyEqual(lMid.value, 5.3, 'AbsTol', 1e-12);
            testCase.verifyEqual(lMid.unit, 'deg');
            testCase.verifyEqual(rMid.value, 3.9, 'AbsTol', 1e-12);
            testCase.verifyEqual(rMid.unit, 'deg');
        end

        function testPermutationInvariance(testCase)
            [keys, units, q] = local_canonical_inputs();
            % Deterministic shuffle
            perm = [25:48, 1:24];
            keys_p = keys(perm);
            units_p = units(perm);
            q_p = q(perm);

            targets1 = gs3dx_initial_target_values(keys, units, q);
            targets2 = gs3dx_initial_target_values(keys_p, units_p, q_p);

            testCase.verifyEqual(targets1.workspace_values, targets2.workspace_values);
            testCase.verifyEqual(targets1.qualification, targets2.qualification);

            % Block position values match regardless of input ordering
            bp1 = targets1.block_position_values;
            bp2 = targets2.block_position_values;
            for i = 1:numel(bp1)
                match = bp2(string({bp2.block}) == string(bp1(i).block) & string({bp2.primitive}) == string(bp1(i).primitive));
                testCase.verifyEqual(bp1(i).value, match.value, 'AbsTol', 1e-12);
            end
        end

        function testRejectionMissingKeys(testCase)
            [keys, units, q] = local_canonical_inputs();
            keys(end) = [];
            units(end) = [];
            q(end) = [];
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q), 'gs3dx:target_values:invalid_keys');
        end

        function testRejectionExtraKeys(testCase)
            [keys, units, q] = local_canonical_inputs();
            keys = [keys; "Extra/Block|Rz.q"];
            units = [units; "deg"];
            q = [q; 0.0];
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q), 'gs3dx:target_values:invalid_keys');
        end

        function testRejectionDuplicateKeys(testCase)
            [keys, units, q] = local_canonical_inputs();
            keys(2) = keys(1);
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q), 'gs3dx:target_values:invalid_keys');
        end

        function testRejectionMalformedOrUnknownKey(testCase)
            [keys, units, q] = local_canonical_inputs();
            keys(1) = "Invalid/Block/PathWithoutPipe";
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q), 'gs3dx:target_values:invalid_keys');
        end

        function testRejectionWrongUnits(testCase)
            [keys, units, q] = local_canonical_inputs();
            % Translation must be 'm', change to 'deg'
            idx = find(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Px.p", 1);
            units(idx) = "deg";
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q), 'gs3dx:target_values:invalid_units');
        end

        function testRejectionInvalidUnitAxis(testCase)
            [keys, units, q] = local_canonical_inputs();
            % Non-zero angle with unnormalized axis
            q(keys == "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_x") = 1.0;
            q(keys == "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_y") = 1.0;
            q(keys == "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_z") = 1.0;
            q(keys == "Lower Body/Right Hip Joint/Kinetically Driven|S.q") = 30.0;
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q), 'gs3dx:target_values:invalid_axis');
        end

        function testIdentityZeroAxisAllowed(testCase)
            [keys, units, q] = local_canonical_inputs();
            % Zero angle with [0, 0, 0] axis must be accepted
            q(keys == "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_x") = 0.0;
            q(keys == "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_y") = 0.0;
            q(keys == "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_z") = 0.0;
            q(keys == "Lower Body/Right Hip Joint/Kinetically Driven|S.q") = 0.0;
            targets = gs3dx_initial_target_values(keys, units, q);
            testCase.verifyEqual(targets.workspace_values.RHipStartPosition, zeros(3, 1), 'AbsTol', 1e-10);
        end

        function testRejectionGimbalSingularity(testCase)
            [keys, units, q] = local_canonical_inputs();
            % 90 deg rotation about Y axis creates middle angle b = 90 deg (singular XYZ)
            q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_x") = 0.0;
            q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_y") = 1.0;
            q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_z") = 0.0;
            q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.q") = 90.0;
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q), 'gs3dx:target_values:gimbal_singularity');
        end

        function testRejectionNonFiniteQ(testCase)
            [keys, units, q] = local_canonical_inputs();
            q(5) = NaN;
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q), 'gs3dx:target_values:invalid_q');
        end

        function testRejectionComplexQ(testCase)
            [keys, units, q] = local_canonical_inputs();
            q(5) = 1.0 + 2.0i;
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q), 'gs3dx:target_values:invalid_q');
        end

        function testRejectionMatrixQ(testCase)
            [keys, units, q] = local_canonical_inputs();
            % 3x16 matrix has 48 elements but is not a vector; must be rejected with invalid_q
            q_matrix = reshape(q, [3, 16]);
            testCase.verifyError(@() gs3dx_initial_target_values(keys, units, q_matrix), 'gs3dx:target_values:invalid_q');
        end
    end
end

function [keys, units, q] = local_canonical_inputs()
    keys = test_gs3dx_initial_target_values.CANONICAL_KEYS;
    n = numel(keys);
    units = strings(n, 1);
    q = zeros(n, 1);

    for i = 1:n
        k = keys(i);
        if contains(k, "|Px.p") || contains(k, "|Py.p") || contains(k, "|Pz.p")
            units(i) = "m";
        elseif contains(k, "|S.ax_")
            units(i) = "1";
        else
            units(i) = "deg";
        end
    end

    % Default nominal pelvis position
    q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Px.p") = 0.1;
    q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Py.p") = -0.2;
    q(keys == "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|Pz.p") = 0.85;

    % Spherical axes default to unit Z with angle 0
    sph_ax_z = [
        "Hips and Torso Inputs/Hip Kinetically Driven/Hip Joint|S.ax_z"
        "Left Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_z"
        "Lower Body/Left Hip Joint/Kinetically Driven|S.ax_z"
        "Lower Body/Right Hip Joint/Kinetically Driven|S.ax_z"
        "Right Shoulder Joint/Gimbal Joint/Kinetically Driven|S.ax_z"
    ];
    for s = sph_ax_z.'
        q(keys == s) = 1.0;
    end
end

function [ax, ang_deg] = local_euler_to_axis_angle_oracle(a_deg, b_deg, c_deg)
% Independent rotation oracle: X-Y-Z Euler -> Matrix -> Axis-Angle
    a = deg2rad(a_deg);
    b = deg2rad(b_deg);
    c = deg2rad(c_deg);

    Rx = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
    Ry = [cos(b) 0 sin(b); 0 1 0; -sin(b) 0 cos(b)];
    Rz = [cos(c) -sin(c) 0; sin(c) cos(c) 0; 0 0 1];
    R = Rx * Ry * Rz;

    tr = trace(R);
    ang_rad = acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)));
    if abs(ang_rad) < 1e-12
        ax = [0.0; 0.0; 1.0];
        ang_deg = 0.0;
    else
        ax = [R(3, 2) - R(2, 3); R(1, 3) - R(3, 1); R(2, 1) - R(1, 2)];
        ax = ax / norm(ax);
        ang_deg = rad2deg(ang_rad);
    end
end
