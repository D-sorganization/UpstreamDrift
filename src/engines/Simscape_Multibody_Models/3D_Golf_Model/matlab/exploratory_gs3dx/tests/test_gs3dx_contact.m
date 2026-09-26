classdef test_gs3dx_contact < matlab.unittest.TestCase
%TEST_GS3DX_CONTACT  GS3DX_FullBodyContact: feet on the ground, free pelvis (#10986).
%
%   Structure (no welds, three sole contacts per foot, an unactuated
%   pelvis, the leg servo), the compiled block budget, and one
%   integration run checked against Newton's second law: with the pelvis
%   drive gone, the foot contacts and gravity must account for the whole
%   change of momentum.

    properties
        info struct
        mdl char
    end

    properties (Constant)
        Body = 'Lower Body'
        Hip = 'Hips and Torso Inputs/Hip Kinetically Driven'
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            testCase.mdl = char(gs3dx_names().variants.contact);
            if ~isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx']))
                gs3dx_build_contact(testCase.info);
            end
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));
        end
    end

    methods (Test)
        function feet_are_not_welded(testCase)
            sys = [testCase.mdl '/' testCase.Body];
            for P = 'LR'
                testCase.verifyEmpty(find_system(sys, 'SearchDepth', 1, 'Name', [P ' Foot Ground']), P);
            end
        end

        function each_foot_has_three_sensed_contacts(testCase)
            sys = [testCase.mdl '/' testCase.Body];
            contacts = find_system(sys, 'SearchDepth', 1, 'Regexp', 'on', 'Name', ' Contact$');
            testCase.verifyNumElements(contacts, 6);
            testCase.verifyEqual(unique(get_param(contacts, 'SenseTotalForce')), {'on'});
            testCase.verifyEqual(unique(get_param(contacts, 'FrictionType')), {'SmoothStickSlip'});
        end

        function pelvis_joint_is_unactuated(testCase)
            hip = [testCase.mdl '/' testCase.Hip];
            joint = [hip '/Hip Joint'];
            for a = {'Px', 'Py', 'Pz', 'Sph'}
                testCase.verifyEqual(get_param(joint, [a{1} 'TorqueActuationMode']), 'NoTorque', a{1});
            end
            testCase.verifyEmpty(find_system(hip, 'SearchDepth', 1, 'Name', 'FollowerTorque'));
        end

        function servo_reference_is_the_stance_hold(testCase)
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            q = ws.getVariable('LegAngleReference');
            testCase.verifySize(q, [12 1]);
            testCase.verifyEqual(ws.getVariable('LHipStartPosition'), q(1:3).', 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.getVariable('RKneeStartPosition'), q(10), 'AbsTol', 1e-12);
            testCase.verifyLessThan(q([4 10]), 0, 'both knees bent forward');
            testCase.verifyEqual(get_param([testCase.mdl '/' testCase.Body '/Leg Torque Commands'], 'Value'), ...
                'LegTorqueCommand + LegServoKp .* LegAngleReference');
        end

        function compiled_budget_leaves_validation_room(testCase)
            b = gs3dx_block_budget(testCase.mdl, compiled=true);
            testCase.verifyLessThanOrEqual(b.compiled_total, gs3dx_names().license_block_limit - 25);
        end

        function body_stands_on_planted_feet_from_rest(testCase)
            c = gs3dx_contact_check(testCase.info, rest=true);
            local_verify_newton(testCase, c);
            for P = 'LR'
                testCase.verifyLessThan(c.feet.(P).slip, 5e-3, [P ' foot slides']);
                testCase.verifyLessThan(c.feet.(P).lift, 1e-3, [P ' foot lifts']);
            end
            testCase.verifyGreaterThan(c.support(2), 0.9, 'the ground carries the body weight');
        end

        function contacts_and_gravity_explain_the_impact_drive_momentum(testCase)
            % The drive's start momentum tips the body over (DATA_AUDIT.md);
            % the momentum balance must still close.
            local_verify_newton(testCase, gs3dx_contact_check(testCase.info));
        end
    end
end

function local_verify_newton(testCase, c)
    testCase.verifyEqual(c.status, "success");
    testCase.verifyLessThanOrEqual(c.newton.max, c.newton.bound, ...
        sprintf('Newton residual %.3g N*s over bound %.3g', c.newton.max, c.newton.bound));
end
