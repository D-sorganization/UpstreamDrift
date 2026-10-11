classdef test_gs3dx_shape < matlab.unittest.TestCase
%TEST_GS3DX_SHAPE  De Leva segment inertia and ellipsoid limbs, GS3DX_Shape (#10979).
%
%   GS3DX_Shape is GS3DX_FitBalance with custom de Leva inertia on the
%   limbs and head and Ellipsoidal Solids for the thighs,
%   shanks, hands and head (GS3DX_BUILD_SHAPE, docs/SHAPE.md).  Its
%   centre-of-mass reference comes from a balance-off run of its own, so
%   this test checks the saved model rather than rebuilding it.

    properties
        info struct
        mdl char
        base char
        audit struct
        base_audit struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            names = gs3dx_names();
            testCase.mdl = char(names.variants.shape);
            testCase.base = char(names.variants.fit_balance);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx'])), ...
                'GS3DX_Shape is built by GS3DX_BUILD_SHAPE (docs/SHAPE.md)');
            for m = {testCase.base testCase.mdl}
                load_system(m{1});
                testCase.addTeardown(@() close_system(m{1}, 0));
            end
            testCase.audit = gs3dx_inertia_audit(testCase.mdl);
            testCase.base_audit = gs3dx_inertia_audit(testCase.base);
        end
    end

    methods (Test)
        function reference_is_an_offset_or_a_path_not_both(testCase)
            testCase.verifyError(@() gs3dx_build_shape(testCase.info, struct(), ...
                com_offset=zeros(3, 2), com_ref=zeros(3, 2)), 'gs3dx:shape');
        end

        function masses_are_unchanged(testCase)
            testCase.verifyEqual(testCase.audit.solids.mass, testCase.base_audit.solids.mass, 'AbsTol', 1e-12);
            testCase.verifyEqual(testCase.audit.total_mass, 80.3929236, 'AbsTol', 1e-7);
        end

        function segment_moments_are_de_levas(testCase)
            seg = testCase.audit.segments;
            changed = ~ismember(seg.segment, ["foot_L" "foot_R" "lower_trunk" "upper_trunk" "trunk"]);
            testCase.verifyEqual(seg.ratio_transverse(changed), ones(nnz(changed), 1), 'AbsTol', 1e-9);
            testCase.verifyEqual(seg.ratio_longitudinal(changed), ones(nnz(changed), 1), 'AbsTol', 1e-9);
            base = testCase.base_audit.segments;
            kept = ~changed;
            testCase.verifyEqual(seg.I_model(kept, :), base.I_model(kept, :), 'AbsTol', 1e-12);
        end

        function centres_of_mass_sit_at_de_levas_fractions(testCase)
            % Proximal joints are at +L/2 on each limb solid's z axis (docs/SHAPE.md).
            a = gs3dx_anthropometry(80);
            ws = get_param(testCase.mdl, 'ModelWorkspace');
            s = testCase.audit.solids;
            com = @(b) s.com(s.block == b, :);
            L = ws.getVariable('ThighLength');
            testCase.verifyEqual(com("Lower Body/L Thigh"), [0 0 L / 2 - a.com.thigh * L], 'AbsTol', 1e-12);
            L = ws.getVariable('ShankLength');
            testCase.verifyEqual(com("Lower Body/R Shank"), [0 0 L / 2 - a.com.shank * L], 'AbsTol', 1e-12);
            % the forearm pair: halves' centres at elbow - c +/- L/4 lump to elbow - c
            L = 0.0254 * double(slResolve('FitLowerArmLength', testCase.mdl));
            zu = L / 4 - com("Left Forearm/LUpperForearm")*[0; 0; 1];
            zl = 3 * L / 4 - com("Left Forearm/LLowerForearm")*[0; 0; 1];
            testCase.verifyEqual((zu + zl) / 2, a.com.forearm * L, 'AbsTol', 1e-12);
            testCase.verifyEqual(zl - zu, L / 2, 'AbsTol', 1e-12);
        end

        function limbs_head_and_hands_are_ellipsoids(testCase)
            for b = ["Lower Body/L Thigh" "Lower Body/R Thigh" "Lower Body/L Shank" "Lower Body/R Shank" ...
                    "Grip/LHand" "Grip/RHand" "Hips and Torso Inputs/Head"]
                blk = [testCase.mdl '/' char(b)];
                testCase.verifyEqual(get_param(blk, 'ReferenceBlock'), 'sm_lib/Body Elements/Ellipsoidal Solid', b);
                testCase.verifyEqual(get_param(blk, 'InertiaType'), 'Custom', b);
                ph = get_param(blk, 'PortHandles');
                testCase.verifyGreaterThan(get_param([ph.LConn ph.RConn], 'Line'), 0, b + " is connected");
            end
        end

        function kinematics_and_drawn_ellipsoids_match(testCase)
            % Same joint targets, same solid poses; drawn ellipsoids are their radii.
            jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
            % Explicitly pin historically Fit-based shape builder test call to variants.fit
            ik = gs3dx_whole_body_ik(jc, model=char(gs3dx_names().variants.fit), ...
                frames=[1 jc.impact_frame], calibration_frames=1:15:jc.impact_frame - 90);
            o = gs3dx_render(testCase.mdl, ik, stills=[1 2], output_dir=tempdir);
            o0 = gs3dx_render(testCase.base, ik, stills=[1 2], output_dir=tempdir);
            load_system(testCase.mdl);
            load_system(testCase.base);
            testCase.verifyEqual(numel(o.solids), numel(o0.solids));
            for i = 1:numel(o.solids)
                j = extractAfter([o0.solids.name], '/') == extractAfter(o.solids(i).name, '/');
                testCase.verifyEqual(o.solids(i).pose.P, o0.solids(j).pose.P, 'AbsTol', 1e-12, o.solids(i).name);
                testCase.verifyEqual(o.solids(i).pose.R, o0.solids(j).pose.R, 'AbsTol', 1e-12, o.solids(i).name);
                if o.solids(i).shape == "Ellipsoid"
                    V = o.solids(i).vertices_local;
                    r = o.solids(i).params.radii(:);
                    testCase.verifyEqual(max(abs(V), [], 2), r, 'AbsTol', 1e-9, o.solids(i).name);
                    testCase.verifyEqual(sum((V ./ r) .^ 2, 1), ones(1, size(V, 2)), 'AbsTol', 1e-9, o.solids(i).name);
                end
            end
            % the head's long axis (z, the upper trunk's) runs along the neck to the head
            head = o.solids(endsWith([o.solids.name], '/Head'));
            neck = o.solids(endsWith([o.solids.name], '/Neck'));
            for f = 1:2
                d = head.pose.P(:, f) - neck.pose.P(:, f);
                testCase.verifyGreaterThan(abs(head.pose.R(:, 3, f)' * d) / norm(d), 0.9);
            end
        end
    end
end
