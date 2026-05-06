classdef test_multistart_parallel_coords < matlab.unittest.TestCase
%TEST_MULTISTART_PARALLEL_COORDS  Smoke tests for MultiStartParallelCoords.
%
%   GitHub issue: #027 / #3996.

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            opt1 = fileparts(here);
            mm   = fileparts(opt1);
            addpath(opt1);
            addpath(fullfile(mm, 'shared'));
            testCase.addTeardown(@() rmpath(opt1));
            testCase.addTeardown(@() rmpath(fullfile(mm, 'shared')));
        end
    end

    methods (Test)
        function test_parallel_coords_renders_n_starts_lines(testCase)
            d = 21; N = 8;
            rng(0);
            result = struct();
            result.start_points = randn(d, N);
            result.start_costs  = rand(1, N);
            opts = struct('visible', false);
            fig = MultiStartParallelCoords.plot(result, opts);
            cleanup = onCleanup(@() delete(fig));
            testCase.verifyTrue(isvalid(fig));
            % Find polylines tagged start_*
            ax = findobj(fig, 'Type', 'Axes');
            lines = findobj(ax, 'Type', 'Line');
            % Count "start_*" lines (should be N) plus best (1)
            tags = string(get(lines, 'Tag'));
            n_start_lines = sum(startsWith(tags, "start_"));
            testCase.verifyEqual(n_start_lines, N);
            n_best = sum(tags == "best_start");
            testCase.verifyEqual(n_best, 1);
        end

        function test_missing_start_points_errors(testCase)
            result = struct('start_costs', [1 2 3]);
            testCase.verifyError( ...
                @() MultiStartParallelCoords.plot(result, struct('visible', false)), ...
                "MultiStartParallelCoords:noStartPoints");
        end

        function test_size_mismatch_errors(testCase)
            result = struct( ...
                'start_points', randn(7, 4), ...
                'start_costs',  [1 2 3]);
            testCase.verifyError( ...
                @() MultiStartParallelCoords.plot(result, struct('visible', false)), ...
                "MultiStartParallelCoords:sizeMismatch");
        end
    end
end
