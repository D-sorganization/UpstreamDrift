classdef test_gs3dx_neck < matlab.unittest.TestCase
%TEST_GS3DX_NECK  GS3DX_Neck: GS3DX_Shape with a motion-driven two-axis neck (#10979).
%
%   GS3DX_Neck is built by GS3DX_BUILD_NECK from GS3DX_Shape and a neck
%   reference measured on the capture (docs/NECK.md), so this test checks
%   the saved model rather than rebuilding it.

    properties
        info struct
        mdl char
        base char
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            names = gs3dx_names();
            testCase.mdl = char(names.variants.neck);
            testCase.base = char(names.variants.shape);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx'])), ...
                'GS3DX_Neck is built by GS3DX_BUILD_NECK (docs/NECK.md)');
            for m = {testCase.base testCase.mdl}
                load_system(m{1});
                testCase.addTeardown(@() close_system(m{1}, 0));
            end
        end
    end

    methods (Test)
        function neck_joint_is_driven_between_trunk_and_neck(testCase)
            sys = [testCase.mdl '/Hips and Torso Inputs'];
            jt = [sys '/Neck Joint'];
            testCase.verifyEqual(strrep(get_param(jt, 'ReferenceBlock'), newline, ' '), 'sm_lib/Joints/Universal Joint');
            f = fieldnames(get_param(jt, 'DialogParameters'));
            for p = f(endsWith(f, 'MotionActuationMode'))'
                testCase.verifyEqual(get_param(jt, p{1}), 'InputMotion', p{1});
            end
            pc = get_param(jt, 'PortConnectivity');
            peers = string(get_param([pc.DstBlock], 'Name'));
            testCase.verifyTrue(any(peers == "Neck"), 'follower joins the Neck');
            testCase.verifyTrue(any(startsWith(peers, "Rigid" + newline + "Transform5")), 'base joins Rigid Transform5');
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            ref = ws.getVariable('NeckReference');
            testCase.verifySize(ref, [2 numel(ws.getVariable('LegReferenceTime'))]);
            testCase.verifyTrue(all(isfinite(ref), 'all'));
        end

        function spheres_go_and_masses_stay(testCase)
            a = gs3dx_inertia_audit(testCase.mdl);
            a0 = gs3dx_inertia_audit(testCase.base);
            testCase.verifyEqual(a.total_mass, a0.total_mass, 'AbsTol', 1e-12);
            gone = setdiff(string(a0.solids.block), string(a.solids.block));
            testCase.verifyEqual(sort(gone), sort(["Left Elbow Joint/Spherical Solid" ...
                "Left Shoulder Joint/Spherical Solid1" "Right Elbow Joint/Spherical Solid1" ...
                "Right Shoulder Joint/Spherical Solid1"]'));
        end

        function compiles_inside_the_reserve(testCase)
            b = gs3dx_block_budget(testCase.mdl, compiled=true);
            names = gs3dx_names();
            testCase.verifyLessThanOrEqual(b.compiled_total, names.license_block_limit - 25);
        end

        function joint_keys_survive_the_added_joint(testCase)
            [k0, id0] = gs3dx_joint_keys(testCase.base);
            [k, id] = gs3dx_joint_keys(testCase.mdl);
            testCase.verifyEqual(sort(setdiff(k, k0)), ...
                ["Hips and Torso Inputs/Neck Joint|Rx.q"; "Hips and Torso Inputs/Neck Joint|Ry.q"]);
            testCase.verifyEmpty(setdiff(k0, k));
            testCase.verifyNotEqual(numel(id), numel(id0));
        end

        function neck_turns_only_neck_and_head(testCase)
            % Straight neck: every solid where GS3DX_Shape has it.  Turned
            % neck: only the Neck and Head move, rigidly about the neck base.
            jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
            ik = gs3dx_whole_body_ik(jc, frames=[1 jc.impact_frame], calibration_frames=1:15:jc.impact_frame - 90);
            [k0, id0] = gs3dx_joint_keys(char(ik.model));
            [~, r] = ismember(string(ik.joint_ids), id0);
            ik.joint_keys = [k0(r); "Hips and Torso Inputs/Neck Joint|Rx.q"; "Hips and Torso Inputs/Neck Joint|Ry.q"];
            ik.joint = [ik.joint; zeros(2, 2)];
            o0 = gs3dx_render(testCase.base, ik, stills=[1 2], output_dir=tempdir);
            o = gs3dx_render(testCase.mdl, ik, stills=[1 2], output_dir=tempdir);
            ik.joint(end - 1:end, :) = [10 -15; -12 5];
            o1 = gs3dx_render(testCase.mdl, ik, stills=[1 2], output_dir=tempdir);
            load_system(testCase.mdl);
            load_system(testCase.base);
            short = @(s) extractAfter([s.name], '/');
            testCase.verifyEqual(numel(o.solids), numel(o0.solids) - 4);
            for i = 1:numel(o.solids)
                j = short(o0.solids) == short(o.solids(i));
                testCase.verifyEqual(o.solids(i).pose.P, o0.solids(j).pose.P, 'AbsTol', 1e-12, o.solids(i).name);
                testCase.verifyEqual(o.solids(i).pose.R, o0.solids(j).pose.R, 'AbsTol', 1e-12, o.solids(i).name);
                moved = max(abs(o1.solids(i).pose.P - o.solids(i).pose.P), [], 'all') > 1e-6;
                testCase.verifyEqual(moved, endsWith(o.solids(i).name, ["/Neck" "/Head"]), o.solids(i).name);
            end
            head = @(s) s.solids(endsWith([s.solids.name], '/Head'));
            neck = @(s) s.solids(endsWith([s.solids.name], '/Neck'));
            for f = 1:2
                % neck base: half a neck below the neck centre, along its z
                base = @(s) neck(s).pose.P(:, f) + neck(s).pose.R(:, 3, f) * 0.127;
                testCase.verifyEqual(base(o1), base(o), 'AbsTol', 1e-9);
                testCase.verifyEqual(norm(head(o1).pose.P(:, f) - base(o1)), norm(head(o).pose.P(:, f) - base(o)), 'AbsTol', 1e-9);
            end
        end
    end
end
