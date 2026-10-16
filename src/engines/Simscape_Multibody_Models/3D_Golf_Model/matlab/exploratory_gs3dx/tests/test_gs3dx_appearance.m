classdef test_gs3dx_appearance < matlab.unittest.TestCase
%TEST_GS3DX_APPEARANCE  GS3DX_Human takes its colours and trunk size from the shared spec (R9c, #12690).
%
%   scripts/export_simscape_appearance.py writes appearance/gs3dx_appearance.v1.json
%   from the shared model_appearance spec; GS3DX_APPLY_APPEARANCE sets it on
%   the saved GS3DX_Human.  Appearance only: the dynamics must not move.

    properties
        info struct
        mdl char
        app struct
        ik struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
            testCase.mdl = char(gs3dx_names().variants.human);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [testCase.mdl '.slx'])), ...
                'GS3DX_Human is built by GS3DX_BUILD_HUMAN (docs/HUMAN.md)');
            testCase.app = gs3dx_read_appearance();
        end
    end

    methods (Test)
        function reader_fails_closed(testCase)
            testCase.verifyError(@() gs3dx_read_appearance(fullfile(tempdir, 'no_such_gs3dx_appearance.json')), ...
                'gs3dx:appearance');
            bad = testCase.app;
            bad.schema_version = 'simscape-appearance/v2';
            f = [tempname '.json'];
            testCase.addTeardown(@() delete(f));
            fid = fopen(f, 'w');
            fwrite(fid, jsonencode(bad));
            fclose(fid);
            testCase.verifyError(@() gs3dx_read_appearance(f), 'gs3dx:appearance');
            bad = testCase.app;
            bad.visuals(1).rgb = [2 0 0];
            fid = fopen(f, 'w');
            fwrite(fid, jsonencode(bad));
            fclose(fid);
            testCase.verifyError(@() gs3dx_read_appearance(f), 'gs3dx:appearance');
        end

        function builder_holds_no_appearance_literals(testCase)
            src = fileread(fullfile(testCase.info.tools_dir, 'gs3dx_build_human.m'));
            body = extractBetween(src, "function vis = local_body_visuals", "function vis = local_club");
            testCase.verifyEmpty(regexp(body{1}, '\b(skin|shirt|pants)\s*=\s*\[', 'once'));
            testCase.verifyNotEmpty(strfind(body{1}, 'gs3dx_read_appearance'));
        end

        function applied_model_equals_the_file_and_the_dynamics_are_unchanged(testCase)
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));
            before = gs3dx_inertia_audit(testCase.mdl);
            applied = gs3dx_apply_appearance(testCase.mdl, testCase.app);
            testCase.verifyNotEmpty(applied);
            for v = testCase.app.visuals(:).'
                blk = applied(endsWith(applied, "/" + string(v.name)));
                testCase.assertNumElements(blk, 1, v.name);
                testCase.verifyEqual(str2num(get_param(blk, 'EllipsoidRadii')), v.radii_m(:).', ...
                    'AbsTol', 1e-9, v.name + " radii"); %#ok<ST2NM>
                testCase.verifyEqual(str2num(get_param(blk, 'GraphicDiffuseColor')), v.rgb(:).', ...
                    'AbsTol', 1e-9, v.name + " colour"); %#ok<ST2NM>
            end
            for r = testCase.app.recolour(:).'
                testCase.verifyEqual(str2num(get_param([testCase.mdl '/' r.block], 'GraphicDiffuseColor')), ...
                    r.rgb(:).', 'AbsTol', 1e-9, r.block); %#ok<ST2NM>
            end
            after = gs3dx_inertia_audit(testCase.mdl);
            testCase.verifyEqual(after.total_mass, before.total_mass, 'AbsTol', 1e-12);
            testCase.verifyEqual(after.solids.mass, before.solids.mass, 'AbsTol', 1e-12);
            testCase.verifyEqual(after.solids.block, before.solids.block);
        end

        function pelvis_wears_the_trousers_and_trunk_is_wider_than_deep_in_the_world(testCase)
            % ML > AP is checked in the Simscape world frame at address, not
            % assumed from the spec: the trunk frames have z down the spine.
            load_system(testCase.mdl);
            testCase.addTeardown(@() close_system(testCase.mdl, 0));
            gs3dx_apply_appearance(testCase.mdl, testCase.app);
            c = testCase.app.colours;
            pelvis = get_param([testCase.mdl '/Hips and Torso Inputs/LowerTorso'], 'Parent');
            pelvis = string(find_system(pelvis, 'SearchDepth', 1, 'Name', 'Pelvis'));
            testCase.verifyEqual(str2num(get_param(pelvis, 'GraphicDiffuseColor')), c.pants(:).', 'AbsTol', 1e-9); %#ok<ST2NM>
            jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
            ik = gs3dx_whole_body_ik(jc, model=testCase.mdl, frames=1, ...
                calibration_frames=1:15:jc.impact_frame - 90);
            o = gs3dx_render(testCase.mdl, ik, stills=1, output_dir=tempdir);
            load_system(testCase.mdl);
            solid = @(n) o.solids(endsWith([o.solids.name], "/" + n));
            extent = @(s, d) norm((s.pose.R(:, :, 1).' * d) .* s.params.radii(:));
            horiz = @(d, spine) (d - spine * (spine.' * d)) / norm(d - spine * (spine.' * d));
            chest = solid("Chest");
            spine = chest.pose.R(:, 3, 1);
            sho = horiz(solid("R Shoulder").pose.P(:, 1) - solid("L Shoulder").pose.P(:, 1), spine);
            nrm = cross(sho, spine) / norm(cross(sho, spine));
            testCase.verifyGreaterThan(extent(chest, sho), extent(chest, nrm), 'chest: shoulder line vs chest normal');
            pel = solid("Pelvis");
            spine = pel.pose.R(:, 3, 1);
            hip = horiz(jc.hip_R(:, 1) - jc.hip_L(:, 1), spine);
            nrm = cross(hip, spine) / norm(cross(hip, spine));
            testCase.verifyGreaterThan(extent(pel, hip), extent(pel, nrm), 'pelvis: hip line vs pelvis normal');
        end
    end
end
