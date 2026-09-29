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
        ik struct
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
                body = below(~startsWith(below, ["Club/" "Grip/"]) & ~endsWith(below, " Sphere"));
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

        function midfoot_joints_bend_the_toes(testCase)
            sys = [testCase.mdl '/Lower Body'];
            for s = ["L" "R"]
                jt = [sys '/' char(s) ' Midfoot Joint'];
                testCase.verifyEqual(strrep(get_param(jt, 'ReferenceBlock'), newline, ' '), 'sm_lib/Joints/Revolute Joint');
                testCase.verifyEqual(get_param(jt, 'SpringStiffness'), 'MidfootStiffness');
                testCase.verifyEqual(get_param(jt, 'MotionActuationMode'), 'ComputedMotion');
                for p = ["Heel In" "Heel Out" "Ball In" "Ball Out" "Toe"]
                    pc = get_param([sys '/' char(s + " " + p + " Point")], 'PortConnectivity');
                    peers = string(strrep(get_param([pc.DstBlock], 'Name'), newline, ' '));
                    testCase.verifyEqual(any(peers == s + " Forefoot Frame"), p == "Toe", ...
                        s + " " + p + ": only the toe rides on the forefoot");
                end
            end
            k = gs3dx_joint_keys(testCase.mdl);
            testCase.verifyTrue(all(ismember(["Lower Body/L Midfoot Joint|Rz.q" "Lower Body/R Midfoot Joint|Rz.q"], k)));
        end

        function five_contacts_per_foot_left_first(testCase)
            % Heel inside/outside, both metatarsal heads, the big toe; the
            % force log keeps the left contacts first (GS3DX_CONTACT_CHECK).
            sys = [testCase.mdl '/Lower Body'];
            testCase.verifyEqual(str2double(get_param([sys '/Foot Contact Mux'], 'Inputs')), 10);
            names = ["Heel In" "Heel Out" "Ball In" "Ball Out" "Toe"];
            k = 0;
            for s = ["L" "R"]
                for n = names
                    k = k + 1;
                    c = get_param([sys sprintf('/Foot Contact Force %d', k)], 'PortConnectivity');
                    src = string(strrep(get_param([c.SrcBlock c.DstBlock], 'Name'), newline, ' '));
                    testCase.verifyTrue(any(src == s + " " + n + " Contact"), sprintf('force %d is %s %s', k, s, n));
                end
            end
        end

        function face_is_square_and_skeleton_unchanged(testCase)
            % Same joint angles: every solid both models draw is where
            % GS3DX_Neck has it (midfoot straight);
            % the face normal points at the target.
            ik = testCase.address_ik();
            o0 = gs3dx_render(testCase.base, ik, stills=1, output_dir=tempdir);
            o = gs3dx_render(testCase.mdl, ik, stills=1, output_dir=tempdir);
            load_system(testCase.mdl);
            load_system(testCase.base);
            short = @(s) extractAfter([s.name], '/');
            % the head and neck carry the neck's address turn (next test)
            shared = setdiff(intersect(short(o0.solids), short(o.solids)), ...
                ["Hips and Torso Inputs/Head" "Hips and Torso Inputs/Neck Shape"]);
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

        function balance_reference_is_anchored_at_this_body(testCase)
            % BalanceCOMRef starts at this model's own address centre of
            % mass (a balance-off start) and is GS3DX_Neck's translated:
            % the head placement moved the COM, not the capture's path.
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            ref = ws.getVariable('BalanceCOMRef');
            d = ref - get_param(testCase.base, 'ModelWorkspace').getVariable('BalanceCOMRef');
            testCase.verifyEqual(d, repmat(d(:, 1), 1, size(d, 2)), 'AbsTol', 1e-12);
            testCase.verifyGreaterThan(norm(d(:, 1)), 1e-3, 'the head placement moves the COM');
            start = ws.getVariable('TrackStart');
            start.BalanceOn = 0;
            c = gs3dx_contact_check(testCase.info, model=testCase.mdl, rest=true, stop_time=1e-3, variables=start);
            load_system(testCase.mdl);
            testCase.verifyEqual(ref(:, 1), c.com(:, 1), 'AbsTol', 1e-6);
        end

        function neck_joins_the_head_and_the_trunk(testCase)
            % The drawn neck runs from inside the head to inside the trunk
            % (chest or trapezius), and the trapezius holds the pivot (C7):
            % no gap shows between the neck and the shoulders.
            o = gs3dx_render(testCase.mdl, testCase.address_ik(), stills=1, output_dir=tempdir);
            load_system(testCase.mdl);
            solid = @(n) o.solids(endsWith([o.solids.name], "/" + n));
            inside = @(s, p) norm((s.pose.R(:, :, 1).' * (p - s.pose.P(:, 1))) ./ s.params.radii(:));
            nk = solid("Neck Shape");
            ends = nk.pose.P(:, 1) + nk.pose.R(:, 3, 1) * [-1 1] * nk.params.radii(3);
            testCase.verifyLessThan(inside(solid("Head"), ends(:, 1)), 1, 'top of the neck in the head');
            testCase.verifyLessThan(min(inside(solid("Chest"), ends(:, 2)), inside(solid("Trapezius"), ends(:, 2))), 1, ...
                'bottom of the neck in the trunk');
            L = slResolve('NeckLength', testCase.mdl) * 0.0254;
            head = solid("Head");
            c7 = head.pose.P(:, 1) + nk.pose.R(:, 3, 1) * L;
            testCase.verifyLessThan(inside(solid("Trapezius"), c7), 1, 'C7 in the trapezius');
        end

        function neck_address_is_geometry_and_the_pivot_is_at_c7(testCase)
            % Human at zero neck angles puts the head where GS3DX_Neck puts
            % it at the NeckAddress angles: the offset is a fixed turn, so
            % the joint starts at its reference.  The pivot moved up the
            % neck and the neck is shorter by the same length.
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            a = ws.getVariable('NeckAddress');
            ik = testCase.address_ik();
            [k0, id0] = gs3dx_joint_keys(char(ik.model));
            [~, r] = ismember(string(ik.joint_ids), id0);
            turned = ik;
            turned.joint_keys = [k0(r); "Hips and Torso Inputs/Neck Joint|Rx.q"; "Hips and Torso Inputs/Neck Joint|Ry.q"];
            turned.joint = [ik.joint; rad2deg(a(:))];
            o0 = gs3dx_render(testCase.base, turned, stills=1, output_dir=tempdir);
            o = gs3dx_render(testCase.mdl, ik, stills=1, output_dir=tempdir);
            load_system(testCase.mdl);
            load_system(testCase.base);
            head = @(o) o.solids(endsWith([o.solids.name], '/Head')).pose.P;
            testCase.verifyEqual(head(o), head(o0), 'AbsTol', 1e-6, 'head centre (m)');
            len = @(m) get_param(m, 'ModelWorkspace').getVariable('NeckLength').Value;
            testCase.verifyEqual(len(testCase.mdl), len(testCase.base) - 0.058 / 0.0254, 'AbsTol', 1e-9);
            nr = @(m) get_param(m, 'ModelWorkspace').getVariable('NeckReference');
            testCase.verifyEqual(nr(testCase.mdl), nr(testCase.base), 'NeckReference is GS3DX_Neck''s');
        end
    end

    methods
        function ik = address_ik(testCase)
            if isempty(testCase.ik)
                jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
                testCase.ik = gs3dx_whole_body_ik(jc, frames=1, calibration_frames=1:15:jc.impact_frame - 90);
            end
            ik = testCase.ik;
        end
    end
end
