classdef test_gs3dx_quiet_reference_profiles < matlab.unittest.TestCase
%TEST_GS3DX_QUIET_REFERENCE_PROFILES  TDD test suite for gs3dx_quiet_reference_profiles (#10979).

    properties (TestParameter)
        bad_time = {0, -1.5, NaN, Inf, [0, 1], "1.0", []}
        bad_qual = {'MAPPING_ONLY', 'SIMULATED', '', 123, ["PURE_MAPPING_ONLY","OTHER"]}
    end

    methods (TestClassSetup)
        function setupClass(~)
            root = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(root,'tools'));
        end
    end

    methods (Test)
        function testIntegerRightAnkleRetainsOtherFractions(testCase)
            [t, expected, ~, ~] = local_analytic_fixture();
            t.workspace_values.LKneeStartPosition = 21.25;
            t.workspace_values.RAnkleStartPosition = int32([-31;-32]);
            expected(4) = 21.25;
            ref = gs3dx_quiet_reference_profiles(t,1);
            testCase.verifyEqual(ref.workspace_values.LegReferenceAngle,repmat(expected,1,2));
        end

        function testMalformedNeckUnitsRejectUniformly(testCase)
            [original, ~, ~, ~] = local_analytic_fixture();
            malformed = {["deg","deg"], [], 17, struct('unit','deg'), string(missing)};
            for k = 1:numel(malformed)
                for row = 1:2
                    t = original;
                    t.block_position_values(row).unit = malformed{k};
                    testCase.verifyError(@() gs3dx_quiet_reference_profiles(t,1), ...
                        'gs3dx:quiet_reference');
                end
            end
        end

        function testMixedNumericLegFieldsRetainFractions(testCase)
            [t, expected, ~, ~] = local_analytic_fixture();
            t.workspace_values.LHipStartPosition = int32([11;12;13]);
            t.workspace_values.LKneeStartPosition = 21.25;
            t.workspace_values.LAnkleStartPosition = [31.5;32.75];
            expected(4:6) = [21.25;31.5;32.75];
            ref = gs3dx_quiet_reference_profiles(t,1);
            testCase.verifyEqual(ref.workspace_values.LegReferenceAngle,repmat(expected,1,2));
        end

        function testNeckMotionProfilesUseRadians(testCase)
            [t, ~, ~, ~] = local_analytic_fixture();
            t.block_position_values(1).value = 90;
            t.block_position_values(2).value = -180;
            ref = gs3dx_quiet_reference_profiles(t,1);
            testCase.verifyEqual(ref.workspace_values.NeckReference, ...
                [pi/2,pi/2;-pi,-pi],'AbsTol',1e-14);
            testCase.verifyEqual(ref.units.NeckReference,'rad');
            testCase.verifyEqual(ref.units.LegReferenceAngle,'deg');
            testCase.verifyEqual(ref.units.LegReferenceRate,'deg/s');
        end

        function testNominalQuietReferenceMapping(testCase)
            [targets, exp_legs, exp_upper, exp_neck] = local_analytic_fixture();
            t_end = 2.5;
            ref = gs3dx_quiet_reference_profiles(targets, t_end);

            % Contractual qualification flags
            testCase.verifyEqual(ref.qualification, 'CONSTANT_REFERENCE_PROFILE_MAPPING_ONLY');
            testCase.verifyEqual(ref.balance_reference_policy, ...
                'REQUIRES_SEPARATE_NATIVE_COM_AND_FOOT_CONFIGURATION');

            ws = ref.workspace_values;
            % Time bases & units (seconds)
            testCase.verifyEqual(ws.LegReferenceTime, [0.0, 2.5]);
            testCase.verifyEqual(ws.UpperBodyTrackTime, [0.0, 2.5]);

            % Leg order [LHip(3); LKnee(1); LAnkle(2); RHip(3); RKnee(1); RAnkle(2)]
            testCase.verifyEqual(size(ws.LegReferenceAngle), [12, 2]);
            testCase.verifyEqual(ws.LegReferenceAngle(:, 1), exp_legs);
            testCase.verifyEqual(ws.LegReferenceAngle(:, 2), exp_legs);
            testCase.verifyEqual(ws.LegReferenceRate, zeros(12, 2));
            testCase.verifyFalse(isfield(ws, 'LegAngleReference'));

            % 12 Upper-body joints cardinality, values, zero rates & torques
            spec = gs3dx_upper_body_joints();
            testCase.verifyNumElements(spec, 12);
            for k = 1:numel(spec)
                pfx = spec(k).prefix;
                n_ax = max(1, numel(spec(k).axes));
                ang = ws.([pfx 'TrackAngle']);
                rate = ws.([pfx 'TrackRate']);
                tau = ws.([pfx 'TrackTorque']);
                testCase.verifyEqual(size(ang), [n_ax, 2]);
                testCase.verifyEqual(size(rate), [n_ax, 2]);
                testCase.verifyEqual(size(tau), [n_ax, 2]);
                testCase.verifyEqual(ang(:, 1), exp_upper.(pfx));
                testCase.verifyEqual(ang(:, 2), exp_upper.(pfx));
                testCase.verifyEqual(rate, zeros(n_ax, 2));
                testCase.verifyEqual(tau, zeros(n_ax, 2));
            end

            % Literal neck reference 2x2
            testCase.verifyEqual(size(ws.NeckReference), [2, 2]);
            testCase.verifyEqual(ws.NeckReference(:, 1), exp_neck);
            testCase.verifyEqual(ws.NeckReference(:, 2), exp_neck);
        end

        function testRejectionBadEndTime(testCase, bad_time)
            [targets, ~, ~, ~] = local_analytic_fixture();
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(targets, bad_time), ...
                'gs3dx:quiet_reference');
        end

        function testRejectionBadQualification(testCase, bad_qual)
            [targets, ~, ~, ~] = local_analytic_fixture();
            targets.qualification = bad_qual;
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(targets, 1.0), ...
                'gs3dx:quiet_reference');
        end

        function testRejectionMalformedTargets(testCase)
            testCase.verifyError(@() gs3dx_quiet_reference_profiles([], 1.0), 'gs3dx:quiet_reference');
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(struct(), 1.0), 'gs3dx:quiet_reference');
        end

        function testRejectionLegFields(testCase)
            [t0, ~, ~, ~] = local_analytic_fixture();
            % Missing field
            t = t0; t.workspace_values = rmfield(t.workspace_values, 'LKneeStartPosition');
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
            % Bad shape
            t = t0; t.workspace_values.LHipStartPosition = [1; 2];
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
            t = t0; t.workspace_values.LKneeStartPosition = [1; 2];
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
            % Non-finite
            t = t0; t.workspace_values.RAnkleStartPosition(1) = NaN;
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
        end

        function testRejectionUpperBodyFields(testCase)
            [t0, ~, ~, ~] = local_analytic_fixture();
            % Missing upper scalar
            t = t0; t.workspace_values = rmfield(t.workspace_values, 'TorsoStartPosition');
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
            % Missing multi-axis component
            t = t0; t.workspace_values = rmfield(t.workspace_values, 'LSStartPositionY');
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
            % Bad shape
            t = t0; t.workspace_values.LEStartPosition = [1, 2];
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
            % Non-finite
            t = t0; t.workspace_values.RSStartPositionZ = Inf;
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
        end

        function testRejectionNeckJointBindings(testCase)
            [t0, ~, ~, ~] = local_analytic_fixture();
            % Missing Rx.q
            t = t0; t.block_position_values(1) = [];
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
            % Duplicate Rx.q
            t = t0; t.block_position_values(2) = t.block_position_values(1);
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
            % Wrong units (rad instead of deg)
            t = t0; t.block_position_values(1).unit = 'rad';
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
            % Non-finite neck value
            t = t0; t.block_position_values(2).value = NaN;
            testCase.verifyError(@() gs3dx_quiet_reference_profiles(t, 1.0), 'gs3dx:quiet_reference');
        end
    end
end

function [targets, exp_legs, exp_upper, exp_neck] = local_analytic_fixture()
    targets.qualification = 'PURE_MAPPING_ONLY';
    ws = struct();
    % Analytic distinct, asymmetric, nonzero leg positions
    ws.LHipStartPosition = [11.0; 12.0; 13.0];
    ws.LKneeStartPosition = 21.0;
    ws.LAnkleStartPosition = [31.0; 32.0];
    ws.RHipStartPosition = [-11.0; -12.0; -13.0];
    ws.RKneeStartPosition = 22.0;
    ws.RAnkleStartPosition = [-31.0; -32.0];
    exp_legs = [11; 12; 13; 21; 31; 32; -11; -12; -13; 22; -31; -32];

    % 12 Upper joints: all nonzero, distinct, asymmetric
    ws.SpineStartPositionX = 1.5; ws.SpineStartPositionY = -2.5; exp_upper.Spine = [1.5; -2.5];
    ws.TorsoStartPosition = 3.5; exp_upper.Torso = 3.5;
    ws.LEStartPosition = 4.5; exp_upper.LE = 4.5;
    ws.LFStartPosition = 5.5; exp_upper.LF = 5.5;
    ws.LScapStartPositionX = 6.5; ws.LScapStartPositionY = 7.5; exp_upper.LScap = [6.5; 7.5];
    ws.LSStartPositionX = 8.5; ws.LSStartPositionY = 9.5; ws.LSStartPositionZ = 10.5; exp_upper.LS = [8.5; 9.5; 10.5];
    ws.LWStartPositionX = 11.5; ws.LWStartPositionY = 12.5; exp_upper.LW = [11.5; 12.5];
    ws.REStartPosition = 13.5; exp_upper.RE = 13.5;
    ws.RFStartPosition = 14.5; exp_upper.RF = 14.5;
    ws.RScapStartPositionX = 15.5; ws.RScapStartPositionY = 16.5; exp_upper.RScap = [15.5; 16.5];
    ws.RSStartPositionX = 17.5; ws.RSStartPositionY = 18.5; ws.RSStartPositionZ = 19.5; exp_upper.RS = [17.5; 18.5; 19.5];
    ws.RWStartPositionX = 20.5; ws.RWStartPositionY = 21.5; exp_upper.RW = [20.5; 21.5];
    targets.workspace_values = ws;

    % Neck joint 2 nonzero literal targets
    targets.block_position_values = [ ...
        struct('block', 'Hips and Torso Inputs/Neck Joint', 'primitive', 'Rx.q', 'value', 7.25, 'unit', 'deg'); ...
        struct('block', 'Hips and Torso Inputs/Neck Joint', 'primitive', 'Ry.q', 'value', -3.75, 'unit', 'deg') ...
    ];
    exp_neck = deg2rad([7.25; -3.75]);
end
