classdef test_gs3dx_ik_points < matlab.unittest.TestCase
%TEST_GS3DX_IK_POINTS  Whole-body IK exposes predicted target points (#11160).

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
        function residual_matches_points_to_capture(testCase)
            fit = char(gs3dx_names().variants.fit);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [fit '.slx'])), 'GS3DX_Fit is not built');
            jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
            off = struct();
            for f = fieldnames(jc.gap).'
                off.(f{1}) = zeros(3, 1);
            end
            ik = gs3dx_whole_body_ik(jc, frames=1, offsets=off, backward=false);
            testCase.verifySize(ik.points, [3, numel(ik.names), numel(ik.frames)]);
            testCase.verifyTrue(any(strcmp(ik.names, 'club_head')));
            f = ik.frames(1);
            for k = 1:numel(ik.names)
                n = ik.names{k};
                r = ik.residual(k, 1);
                if ~isnan(r)
                    d = norm(ik.points(:, k, 1) - jc.(n)(:, f));
                    testCase.verifyEqual(r, d, 'AbsTol', 1e-12, n);
                end
            end
        end

        function target_weight_default_and_club_head(testCase)
            fit = char(gs3dx_names().variants.fit);
            testCase.assumeTrue(isfile(fullfile(testCase.info.models_dir, [fit '.slx'])), 'GS3DX_Fit is not built');
            jc = gs3dx_capture_joint_centres(gs3dx_capture_markers());
            off = struct();
            for f = fieldnames(jc.gap).'
                off.(f{1}) = zeros(3, 1);
            end
            base = {'frames', 1, 'offsets', off, 'backward', false};
            ik0 = gs3dx_whole_body_ik(jc, base{:});
            tw1 = ones(1, numel(ik0.names));
            ik1 = gs3dx_whole_body_ik(jc, base{:}, 'target_weight', tw1);
            testCase.verifyEqual(ik1.residual, ik0.residual, 'AbsTol', 1e-10);
            testCase.verifyEqual(ik1.points, ik0.points, 'AbsTol', 1e-10);

            tw = ones(1, numel(ik0.names));
            ih = find(strcmp(ik0.names, 'club_head'), 1);
            tw(ih) = 0;
            ik2 = gs3dx_whole_body_ik(jc, base{:}, 'target_weight', tw);
            moved = ~isequal(ik2.residual(ih, :), ik0.residual(ih, :)) || ...
                any(abs(ik2.points(:, ih, 1) - ik0.points(:, ih, 1)) > 1e-10);
            if ~moved
                testCase.verifyEqual(ik2.residual, ik0.residual, 'AbsTol', 1e-10);
                testCase.verifyEqual(ik2.points, ik0.points, 'AbsTol', 1e-10);
            end
        end
    end
end
