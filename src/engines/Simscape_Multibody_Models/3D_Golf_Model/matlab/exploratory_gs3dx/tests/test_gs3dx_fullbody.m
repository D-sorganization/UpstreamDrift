classdef test_gs3dx_fullbody < matlab.unittest.TestCase
%TEST_GS3DX_FULLBODY  GS3DX_FullBody: GS3DX_Quat plus legs and welded feet (#10957, #10958).
%
%   Structure (two of each leg joint, feet framed to World), the block
%   budget with a 10% margin, and an integration run: the full body starts
%   in exactly GS3DX_Quat's state, the legs assemble on their start
%   targets, and the swing window completes.

    properties
        info struct
        names struct
    end

    properties (Constant)
        Body = 'Lower Body'
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info  = gs3dx_setup();
            testCase.names = gs3dx_names();
            full = char(testCase.names.variants.fullbody);
            if ~isfile(fullfile(testCase.info.models_dir, [full '.slx']))
                gs3dx_build_lower_body(testCase.info);
            end
        end
    end

    methods (Test)
        function lower_body_has_two_of_each_leg_joint(testCase)
            sys = local_body(testCase);
            for j = gs3dx_leg_table().joints
                refs = find_system(sys, 'SearchDepth', 1, 'LookInsideSubsystemReference', 'off', ...
                    'BlockType', 'SubSystem', 'ReferencedSubsystem', j.kds);
                testCase.verifyEqual(sort(get_param(refs, 'Name')), ...
                    sort({['Left ' j.name ' Joint']; ['Right ' j.name ' Joint']}), j.kds);
            end
        end

        function feet_are_framed_to_world(testCase)
            sys = local_body(testCase);
            for side = 'LR'
                pc = get_param([sys '/' side ' Foot Ground'], 'PortConnectivity');
                base = [pc(strcmp({pc.Type}, 'LConn1')).DstBlock];
                testCase.verifyTrue(any(strcmp(get_param(base, 'Name'), 'World')), side);
            end
        end

        function leg_frame_is_a_rotation_with_z_up(testCase)
            local_body(testCase);
            ws = get_param(char(testCase.names.variants.fullbody), 'ModelWorkspace');
            R = ws.getVariable('LFootGroundRotation');
            testCase.verifyEqual(R.' * R, eye(3), 'AbsTol', 1e-12);
            testCase.verifyEqual(det(R), 1, 'AbsTol', 1e-12);
            testCase.verifyEqual(R(:, 3), [0; 0; 1], 'AbsTol', 1e-12);
            testCase.verifyEqual(ws.getVariable('RFootGroundRotation'), R);
        end

        function fullbody_keeps_a_ten_percent_block_margin(testCase)
            full = char(testCase.names.variants.fullbody);
            if bdIsLoaded(full)
                close_system(full, 0);
            end
            report = gs3dx_block_budget(full);
            testCase.verifyLessThanOrEqual(report.nonvirtual_total, 0.9 * testCase.names.license_block_limit);
            fprintf('GS3DX_FullBody: %d non-virtual blocks\n', report.nonvirtual_total);
        end

        function builder_refuses_to_overwrite(testCase)
            testCase.verifyError(@() gs3dx_build_lower_body(testCase.info), 'gs3dx:fullbody');
        end
    end

    methods (Test, TestTags = {'Simulation'})
        function fullbody_starts_in_the_quat_state_and_completes_the_window(testCase)
            quat = char(testCase.names.variants.quat);
            full = char(testCase.names.variants.fullbody);
            q = local_run(testCase.info, quat);
            f = local_run(testCase.info, full);
            testCase.assertEqual(f.status, "success", f.message);
            for key = ["CHGlobalPosition", "MPGlobalPosition", "HipGlobalPosition"]
                k = find(endsWith(q.flat.names, key), 1);
                testCase.verifyLessThanOrEqual(max(abs(f.flat.data{k}(1, :) - q.flat.data{k}(1, :))), 1e-9, key);
            end
            % Knee targets are Low priority and the table's start angles are
            % only nearly consistent (thigh and shank lengths differ), so
            % assembly lands within ~1e-3 deg; 0.01 deg confirms the
            % bent-forward branch.
            leg = gs3dx_leg_table();
            for side = 'LR'
                knee = f.logsout.get([side 'KneeLogs']).Values.AngularPosition.Data;
                testCase.verifyEqual(knee(1), -2 * leg.theta, 'AbsTol', 0.01, [side ' knee start']);
            end
            fprintf('GS3DX_FullBody: %d steps, %.1f s wall (GS3DX_Quat %d, %.1f s)\n', ...
                f.n_steps, f.wall_s, q.n_steps, q.wall_s);
        end
    end
end

function sys = local_body(testCase)
    full = char(testCase.names.variants.fullbody);
    load_system(full);
    sys = [full '/' testCase.Body];
end

function run = local_run(info, mdl)
    load_system(mdl);
    run = gs3dx_simulate(mdl, variables = gs3dx_drive(info, "impact", mdl));
end
