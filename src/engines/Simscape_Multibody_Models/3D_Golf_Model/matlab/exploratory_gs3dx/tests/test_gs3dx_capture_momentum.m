classdef test_gs3dx_capture_momentum < matlab.unittest.TestCase
%TEST_GS3DX_CAPTURE_MOMENTUM  Capture segment masses and angular momentum (#11173).
%
%   Runs on the tour-average driver capture; skipped when MATLAB's Python
%   has no ezc3d (the capture reader needs it).

    methods (TestClassSetup)
        function setup(testCase)
            addpath(fileparts(fileparts(mfilename('fullpath'))));
            gs3dx_setup();
            try
                py.importlib.import_module('ezc3d');
                has_ezc3d = true;
            catch
                has_ezc3d = false;
            end
            testCase.assumeTrue(has_ezc3d, 'Python ezc3d is not available to MATLAB (pyenv)');
        end
    end

    methods (Test)
        function segment_masses_sum_to_body_plus_club(testCase)
            [mass, pos] = gs3dx_capture_segments(body_mass=80, club_head_mass=0.25, club_rest_mass=0.137);
            testCase.verifySize(pos, [3 numel(mass) size(pos, 3)]);
            testCase.verifyEqual(sum(mass), 80 + 0.25 + 0.137, 'AbsTol', 0.5);
            testCase.verifyTrue(all(mass > 0));
        end
        function momentum_is_finite_and_consistent_in_size(testCase)
            q = gs3dx_capture_momentum();
            testCase.verifySize(q.h, [3 numel(q.t)]);
            testCase.verifySize(q.hdot, [3 numel(q.t)]);
            testCase.verifyTrue(all(isfinite(q.h(:))));
        end
        function hdot_matches_the_derivative_of_h_away_from_the_ends(testCase)
            q = gs3dx_capture_momentum();
            dt = q.t(2) - q.t(1);
            num = gradient(q.h, dt);
            mid = 20:numel(q.t) - 20;
            scale = max(abs(q.hdot(:)));
            testCase.verifyLessThan(max(abs(num(:, mid) - q.hdot(:, mid)), [], 'all'), 0.05 * scale);
        end
    end
end
