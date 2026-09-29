classdef test_progress_dashboard < matlab.unittest.TestCase
%TEST_PROGRESS_DASHBOARD  Headless smoke tests for OptimizationProgressDashboard.
%
%   Tests construct the dashboard with Visible=off and push synthetic
%   iteration records; they do not run any optimization.
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

    methods (Static, Access = private)
        function opts = headless_opts()
            opts = OptimizationProgressDashboard.default_options();
            opts.visible          = false;
            opts.autostart_timer  = false;  % deterministic for tests
        end

        function ov = mk_optim_values(iter, fval, grad, step)
            ov = struct( ...
                'iteration',     iter, ...
                'fval',          fval, ...
                'firstorderopt', grad, ...
                'stepsize',      step);
        end
    end

    methods (Test)
        function test_dashboard_constructs_four_panels(testCase)
            target = struct('name', "synthetic");
            dash = OptimizationProgressDashboard(target, ...
                test_progress_dashboard.headless_opts());
            cleanup = onCleanup(@() dash.close());
            testCase.verifyTrue(isvalid(dash.Figure));
            testCase.verifyNotEmpty(dash.AxCost);
            testCase.verifyNotEmpty(dash.AxGrad);
            testCase.verifyNotEmpty(dash.AxStep);
            testCase.verifyNotEmpty(dash.AxTheta);
        end

        function test_outputfcn_pushes_values_to_queue(testCase)
            dash = OptimizationProgressDashboard(struct(), ...
                test_progress_dashboard.headless_opts());
            cleanup = onCleanup(@() dash.close());
            fcn = dash.outputFcn();
            testCase.verifyClass(fcn, 'function_handle');
            stop = fcn(zeros(5,1), ...
                test_progress_dashboard.mk_optim_values(1, 10.0, 1.0, 0.1), ...
                "iter");
            testCase.verifyFalse(stop);
            testCase.verifyEqual(dash.IterationCount, 1);
            % Drain so renderPanels can run
            dash.drainAndRender();
            testCase.verifyEqual(numel(dash.History.iter), 1);
            testCase.verifyEqual(dash.History.fval(1), 10.0);
        end

        function test_dashboard_drops_updates_when_busy(testCase)
            % 100 pushes, but with autostart_timer=false we manually call
            % drainAndRender once, so RedrawCount must be 1, not 100.
            dash = OptimizationProgressDashboard(struct(), ...
                test_progress_dashboard.headless_opts());
            cleanup = onCleanup(@() dash.close());
            fcn = dash.outputFcn();
            for k = 1:100
                fcn(rand(3,1), ...
                    test_progress_dashboard.mk_optim_values(k, 1/k, 1/k, 1/k), ...
                    "iter");
            end
            testCase.verifyEqual(dash.IterationCount, 100);
            % Simulate <=5 timer ticks (1s @ 5Hz)
            for k = 1:5
                dash.drainAndRender();
            end
            testCase.verifyLessThanOrEqual(dash.RedrawCount, 5);
            % All 100 records folded into history (within history_limit)
            testCase.verifyEqual(numel(dash.History.iter), 100);
        end

        function test_close_cleans_up_timer_and_queue(testCase)
            dash = OptimizationProgressDashboard(struct(), ...
                test_progress_dashboard.headless_opts());
            fig = dash.Figure;
            timer_obj = dash.RefreshTimer;
            testCase.verifyTrue(isvalid(timer_obj));
            dash.close();
            testCase.verifyTrue(dash.Closed);
            testCase.verifyEmpty(dash.RefreshTimer);
            testCase.verifyFalse(isvalid(fig));
            % Idempotent
            dash.close();
        end

        function test_dashboard_input_validation(testCase)
            bad = struct();
            bad.refresh_hz = -1;
            testCase.verifyError( ...
                @() OptimizationProgressDashboard(struct(), bad), ...
                "OptimizationProgressDashboard:badRefresh");
            bad2 = struct();
            bad2.history_limit = -10;
            testCase.verifyError( ...
                @() OptimizationProgressDashboard(struct(), bad2), ...
                "OptimizationProgressDashboard:badHistoryLimit");
        end

        function test_history_limit_caps_memory(testCase)
            opts = test_progress_dashboard.headless_opts();
            opts.history_limit = 10;
            dash = OptimizationProgressDashboard(struct(), opts);
            cleanup = onCleanup(@() dash.close());
            fcn = dash.outputFcn();
            for k = 1:50
                fcn(zeros(2,1), ...
                    test_progress_dashboard.mk_optim_values(k, 1/k, NaN, NaN), ...
                    "iter");
            end
            dash.drainAndRender();
            testCase.verifyLessThanOrEqual(numel(dash.History.iter), 10);
        end

        function test_outputfcn_compatible_with_fmincon_signature(testCase)
            % fmincon passes char state, real x, real optimValues
            dash = OptimizationProgressDashboard(struct(), ...
                test_progress_dashboard.headless_opts());
            cleanup = onCleanup(@() dash.close());
            fcn = dash.outputFcn();
            ov = struct('iteration', 1, 'fval', 0.5, ...
                'firstorderopt', 0.01, 'stepsize', 0.1);
            stop = fcn([1;2;3], ov, 'iter');   % char, not string
            testCase.verifyFalse(stop);
        end
    end
end
