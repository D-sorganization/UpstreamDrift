classdef test_gs3dx_render_grounded < matlab.unittest.TestCase
%TEST_GS3DX_RENDER_GROUNDED  Regression test for fixed-foot rendering in GS3DX_FullBody (#11329).

    methods (TestClassSetup)
        function setupEnvironment(tc)
            here = fileparts(mfilename('fullpath'));
            tc.applyFixture(matlab.unittest.fixtures.PathFixture(here));
            root=fileparts(here);
            if isfile(fullfile(root,'gs3dx_setup.m'))
                tc.applyFixture(matlab.unittest.fixtures.PathFixture(root));
            end

            tc.assumeTrue(~isempty(which('gs3dx_setup')), 'gs3dx_setup unavailable.');
            gs3dx_setup();

            names = gs3dx_names();
            full_model = char(names.variants.fullbody);
            tc.assumeTrue(isfile(which([full_model '.slx'])), ...
                sprintf('Model %s.slx not found on path.', full_model));

            has_sm = exist('simscape.multibody.KinematicsSolver', 'class') == 8;
            tc.assumeTrue(has_sm, 'Native Simscape Multibody KinematicsSolver unavailable.');
        end
    end

    methods (Test)
        function testGroundedFeetRenderInvariant(tc)
            names = gs3dx_names();
            mdl = char(names.variants.fullbody);

            if ~bdIsLoaded(mdl)
                load_system(mdl);
            end
            tc.addTeardown(@() test_gs3dx_render_grounded.closeModel(mdl));

            % 1. Target-free native solve to obtain one valid pose
            try
                ks_free = simscape.multibody.KinematicsSolver(mdl);
            catch err
                if contains(lower(err.message),'license')
                    tc.assumeFail(sprintf('Native license unavailable: %s',err.message));
                else
                    rethrow(err);
                end
            end
            cleanup_free = onCleanup(@() delete(ks_free)); %#ok<NASGU>
            jp = ks_free.jointPositionVariables;
            [~, all_ids] = gs3dx_joint_keys(mdl, jp);
            addOutputVariables(ks_free, all_ids);

            [q0, st0] = solve(ks_free, [], []);
            tc.assertEqual(st0, 1, 'Native target-free KinematicsSolver failed to solve base pose.');
            clear cleanup_free;

            % 2. Build shared grounded roles solver
            roles = gs3dx_ik_joint_roles(jp.BlockPath, all_ids, 'grounded_legs', true);
            targets = roles.target_ids;
            guesses = roles.closed_ids;

            ks_grounded = simscape.multibody.KinematicsSolver(mdl);
            cleanup_grounded = onCleanup(@() delete(ks_grounded)); %#ok<NASGU>
            addTargetVariables(ks_grounded, targets);
            addInitialGuessVariables(ks_grounded, guesses);
            addOutputVariables(ks_grounded, all_ids);

            % 3. Change upper-body pose without crossing the rigid-leg reach boundary
            [~, tgt_loc] = ismember(targets, all_ids);
            [~, guess_loc] = ismember(guesses, all_ids);
            tgt_vals = q0(tgt_loc);
            guess_vals = q0(guess_loc);

            torso_id=string(jp.ID(contains(string(jp.BlockPath),'Torso Kinetically Driven') & endsWith(string(jp.ID),'.Rz.q')));
            tc.assertNumElements(torso_id,1);
            idx_torso=find(targets==torso_id);tc.assertNumElements(idx_torso,1);
            tgt_vals(idx_torso)=tgt_vals(idx_torso)+12;

            [q1, st1] = solve(ks_grounded, tgt_vals, guess_vals);
            tc.assertEqual(st1, 1, 'KinematicsSolver failed on perturbed grounded pose (status ~= 1).');
            clear cleanup_grounded;

            % Independently derive fixed-foot transforms before renderer closes the model
            ws = get_param(mdl, 'ModelWorkspace');
            foot_length = ws.getVariable('FootLength');
            foot_heel_offset = ws.getVariable('FootHeelOffset');
            ankle_height = ws.getVariable('AnkleHeight');
            local_com_offset = [(0.5 - foot_heel_offset) * foot_length; 0; -ankle_height / 2];

            expected_foot_R=zeros(3,3,2);expected_foot_p=zeros(3,2);
            sides=["L","R"];
            for i=1:2
                expected_foot_R(:,:,i)=ws.getVariable(char(sides(i)+"FootGroundRotation"));
                expected_foot_p(:,i)=ws.getVariable(char(sides(i)+"FootGroundOffset"))+expected_foot_R(:,:,i)*local_com_offset;
            end

            % 4. Render the two poses via gs3dx_render to invisible PNG stills
            temp_fixture = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            out_dir = temp_fixture.Folder;

            q_ik = struct('model', mdl, 'joint_ids', all_ids, 'joint', [q0(:), q1(:)]);
            still_files = ["pose1.png", "pose2.png"];
            render_out = gs3dx_render(mdl, q_ik, ...
                'stills', [1 2], ...
                'still_files', still_files, ...
                'output_dir', out_dir, ...
                'resolution', [1920 1080], ...
                'visible', false);

            sides = ["L", "R"];
            for i=1:2
                s=sides(i);expected_R=expected_foot_R(:,:,i);expected_p=expected_foot_p(:,i);

                foot_blk = string(mdl) + "/Lower Body/" + s + " Foot";
                idx_solid = find(string({render_out.solids.block}) == foot_blk);
                tc.assertNumElements(idx_solid, 1, sprintf('Solid block for %s Foot not found.', s));

                foot_solid = render_out.solids(idx_solid);
                for f = 1:2
                    tc.verifyEqual(foot_solid.pose.R(:, :, f), expected_R, 'AbsTol', 1e-7, ...
                        sprintf('%s Foot pose.R mismatch on frame %d', s, f));
                    tc.verifyEqual(foot_solid.pose.P(:, f), expected_p, 'AbsTol', 1e-7, ...
                        sprintf('%s Foot pose.P mismatch on frame %d', s, f));
                end
            end

            % 6. Assert two PNGs exist and are 1920x1080
            for sf = still_files
                png_path = fullfile(out_dir, char(sf));
                tc.verifyTrue(isfile(png_path), sprintf('Still image %s does not exist.', sf));
                img_info = imfinfo(png_path);
                tc.verifyEqual(img_info.Width, 1920, sprintf('%s width is not 1920.', sf));
                tc.verifyEqual(img_info.Height, 1080, sprintf('%s height is not 1080.', sf));
            end
        end
    end
    methods (Static, Access=private)
        function closeModel(mdl)
            if bdIsLoaded(mdl),close_system(mdl,0);end
        end
    end
end
