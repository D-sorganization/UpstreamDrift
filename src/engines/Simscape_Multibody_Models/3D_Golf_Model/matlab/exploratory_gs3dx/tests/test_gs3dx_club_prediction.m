classdef test_gs3dx_club_prediction < matlab.unittest.TestCase
%TEST_GS3DX_CLUB_PREDICTION  Club track from whole-body IK (#11160).

    properties
        info struct
    end

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            testCase.info = gs3dx_setup();
        end
    end

    methods (Test)
        function frame_one_has_finite_club_geometry(testCase)
            fit = char(gs3dx_names().variants.fit);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [fit '.slx'])), 'GS3DX_Fit is not built');
            cap = gs3dx_capture_markers();
            jc = gs3dx_capture_joint_centres(cap);
            off = struct();
            for f = fieldnames(jc.gap).'
                off.(f{1}) = zeros(3, 1);
            end
            ik = gs3dx_whole_body_ik(jc, frames=1, offsets=off, backward=false);
            pred = gs3dx_club_prediction(ik, jc, cap);
            ih = find(strcmp(ik.names, 'club_head'), 1);
            testCase.verifyEqual(pred.head, ik.points(:, ih, 1), 'AbsTol', 1e-12);
            testCase.verifySize(pred.face_normal, [3 1]);
            testCase.verifyEqual(norm(pred.face_normal), 1, 'AbsTol', 1e-9);
            testCase.verifySize(pred.grip, [3 1]);
            testCase.verifyTrue(all(isfinite(pred.grip)));
        end

        function the_club_matches_the_capture(testCase)
            fit = char(gs3dx_names().variants.fit);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [fit '.slx'])), 'GS3DX_Fit is not built');
            cap = gs3dx_capture_markers();
            jc = gs3dx_capture_joint_centres(cap);
            f0 = jc.impact_frame;
            frames = unique([1:10:f0 f0]);
            reg = {'posture_weight', 0.01, 'smooth_weight', 0.02, 'backward', false, 'gap_weight', 0.5};
            off = gs3dx_whole_body_ik(jc, frames=1, calibration_frames=1:15:f0 - 90).offsets;
            ik = gs3dx_whole_body_ik(jc, 'frames', frames, 'offsets', off, reg{:});
            pred = gs3dx_club_prediction(ik, jc, cap);

            ref.head = jc.club_head(:, ik.frames);
            ref.grip = jc.club_grip(:, ik.frames);
            ref.face_normal = pred.ref_face_normal;
            sim.head = pred.head;
            sim.grip = pred.grip;
            sim.face_normal = pred.face_normal;

            head_z = ref.head(3, :);
            [~, p] = max(head_z(2:end - 1));
            phases = struct('address', 1, 'top', p + 1, 'impact', numel(ik.frames));
            M = gs3dx_club_match(ref, sim, ik.t, phases);
            if ~M.ok.all
                a = M.phase(M.phase.phase == "address_to_impact", :);
                testCase.log(1, char(formattedDisplayText(a)));
                chk = M.ok;
                fail = fieldnames(chk);
                fail = fail(~structfun(@(x) x, chk));
                testCase.log(1, sprintf('Failing ok checks: %s', strjoin(fail, ', ')));
            end
            testCase.verifyTrue(M.ok.all);
        end
    end
end
