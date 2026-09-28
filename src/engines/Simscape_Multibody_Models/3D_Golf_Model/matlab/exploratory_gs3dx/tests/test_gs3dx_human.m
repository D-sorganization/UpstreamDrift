classdef test_gs3dx_human < matlab.unittest.TestCase
%TEST_GS3DX_HUMAN  GS3DX_Human: ellipsoid body, square face and jointed feet (#10979).
%
%   GS3DX_Human is built by GS3DX_BUILD_HUMAN from GS3DX_Neck
%   (docs/HUMAN.md), so this test checks the saved model rather than
%   rebuilding it.

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
            testCase.mdl = char(names.variants.human);
            testCase.base = char(names.variants.neck);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx'])), ...
                'GS3DX_Human is built by GS3DX_BUILD_HUMAN (docs/HUMAN.md)');
            for m = {testCase.base testCase.mdl}
                load_system(m{1});
                testCase.addTeardown(@() close_system(m{1}, 0));
            end
        end
    end

    methods (Test)
        function body_is_drawn_with_ellipsoids(testCase)
            % Every visible solid off the club is an ellipsoid, bar the
            % 1 cm foot contact spheres.
            refs = ["Brick Solid" "Cylindrical Solid" "Spherical Solid" "Ellipsoidal Solid"];
            for r = refs
                blks = string(find_system(testCase.mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
                    'ReferenceBlock', char("sm_lib/Body Elements/" + r)));
                blks = blks(get_param(cellstr(blks), 'GraphicType') ~= "None");
                below = extractAfter(blks, strlength(testCase.mdl) + 1);
                if r == "Ellipsoidal Solid"
                    continue
                end
                body = below(~startsWith(below, ["Club/" "Grip/"]) & ~contains(below, ["Heel Sphere" "Toe In Sphere" "Toe Out Sphere"]));
                testCase.verifyEmpty(body, r + " drawn on the body");
            end
            testCase.verifyEqual(get_param([testCase.mdl '/Hips and Torso Inputs/ZeroMassHipReference'], 'GraphicType'), 'None');
        end

        function mass_stays_and_the_forefoot_takes_its_share(testCase)
            a = gs3dx_inertia_audit(testCase.mdl);
            a0 = gs3dx_inertia_audit(testCase.base);
            testCase.verifyEqual(a.total_mass, a0.total_mass, 'AbsTol', 1e-9);
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            mf = ws.getVariable('ForefootMass');
            m = @(b) a.solids.mass(string(a.solids.block) == b);
            m0 = @(b) a0.solids.mass(string(a0.solids.block) == b);
            for s = ["L" "R"]
                testCase.verifyEqual(m("Lower Body/" + s + " Forefoot"), mf, 'AbsTol', 1e-12);
                testCase.verifyEqual(m("Lower Body/" + s + " Foot") + mf, m0("Lower Body/" + s + " Foot"), 'AbsTol', 1e-12);
            end
        end

        function sensor_subsystem_goes_and_budget_holds(testCase)
            testCase.verifyEmpty(find_system(testCase.mdl, 'SearchDepth', 1, 'Name', 'Inertia Sensor'));
            b = gs3dx_block_budget(testCase.mdl, compiled=true);
            names = gs3dx_names();
            testCase.verifyLessThanOrEqual(b.compiled_total, names.license_block_limit - 25);
        end

        function midfoot_joints_carry_the_toes(testCase)
            sys = [testCase.mdl '/Lower Body'];
            for s = ["L" "R"]
                jt = [sys '/' char(s) ' Midfoot Joint'];
                testCase.verifyEqual(strrep(get_param(jt, 'ReferenceBlock'), newline, ' '), 'sm_lib/Joints/Revolute Joint');
                testCase.verifyEqual(get_param(jt, 'SpringStiffness'), 'MidfootStiffness');
                testCase.verifyEqual(get_param(jt, 'MotionActuationMode'), 'ComputedMotion');
                for p = ["Toe In Point" "Toe Out Point"]
                    pc = get_param([sys '/' char(s + " " + p)], 'PortConnectivity');
                    peers = string(strrep(get_param([pc.DstBlock], 'Name'), newline, ' '));
                    testCase.verifyTrue(any(peers == s + " Forefoot Frame"), s + " " + p + " rides on the forefoot");
                end
            end
            k = gs3dx_joint_keys(testCase.mdl);
            testCase.verifyTrue(all(ismember(["Lower Body/L Midfoot Joint|Rz.q" "Lower Body/R Midfoot Joint|Rz.q"], k)));
        end

        function face_is_square_and_skeleton_unchanged(testCase)
            % Same joint angles: every solid both models draw is where
            % GS3DX_Neck has it, the toe spheres too (midfoot straight);
            % the face normal points at the target.
            jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
            ik = gs3dx_whole_body_ik(jc, frames=1, calibration_frames=1:15:jc.impact_frame - 90);
            o0 = gs3dx_render(testCase.base, ik, stills=1, output_dir=tempdir);
            o = gs3dx_render(testCase.mdl, ik, stills=1, output_dir=tempdir);
            load_system(testCase.mdl);
            load_system(testCase.base);
            short = @(s) extractAfter([s.name], '/');
            shared = intersect(short(o0.solids), short(o.solids));
            testCase.verifyGreaterThan(numel(shared), 10);
            for n = shared
                a = o0.solids(short(o0.solids) == n);
                b = o.solids(short(o.solids) == n);
                testCase.verifyEqual(b.pose.P, a.pose.P, 'AbsTol', 1e-9, n);
            end
            fn = o.solids(endsWith([o.solids.name], '/Face Normal'));
            v = fn.pose.R(:, 3, 1);
            testCase.verifyLessThan(abs(atan2d(v(1), v(2))), 1, 'face angle at address (deg)');
            testCase.verifyGreaterThan(asind(v(3)), 5, 'loft shows');
        end
    end
end
