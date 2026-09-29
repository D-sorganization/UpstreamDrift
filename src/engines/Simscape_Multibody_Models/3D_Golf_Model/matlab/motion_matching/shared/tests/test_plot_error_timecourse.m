classdef test_plot_error_timecourse < matlab.unittest.TestCase
%TEST_PLOT_ERROR_TIMECOURSE  Tests for the View 2 stacked plot.
%   See VISUALIZATION_SPEC.md for the styling and layout contract.

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            shared = fullfile(here, '..');
            addpath(shared);
            testCase.addTeardown(@() rmpath(shared));
        end
    end

    methods (Test)
        function test_returns_figure_with_four_subplots(testCase)
            [res, tgt] = make_pair();
            fig = plot_error_timecourse(res, tgt, headless_opts());
            testCase.addTeardown(@() close(fig));
            testCase.verifyClass(fig, 'matlab.ui.Figure');
            ax = findobj(fig, 'Type', 'axes');
            testCase.verifyEqual(numel(ax), 4);
        end

        function test_default_opts_contract(testCase)
            opts = default_viz_options();
            testCase.verifyTrue(isfield(opts, 'figure_visible'));
            testCase.verifyTrue(isfield(opts, 'sample_rate_noise_m'));
            testCase.verifyTrue(isfield(opts, 'dpi'));
            testCase.verifyEqual(opts.sample_rate_noise_m, 5e-4);
        end

        function test_joint_torques_one_trace_per_joint(testCase)
            [res, tgt] = make_pair();
            % set torques to 5 distinct joints
            n_joints = 5;
            N = numel(tgt.time);
            res.sim_out.tau = repmat((1:n_joints), N, 1) .* (1:N).';
            fig = plot_error_timecourse(res, tgt, headless_opts());
            testCase.addTeardown(@() close(fig));
            ax = local_axes_with_ylabel( ...
                findobj(fig, 'Type', 'axes'), 'Joint torque');
            lines = findobj(ax, 'Type', 'line');
            testCase.verifyEqual(numel(lines), n_joints);
        end

        function test_impact_line_present_in_all_panels(testCase)
            [res, tgt] = make_pair();
            fig = plot_error_timecourse(res, tgt, headless_opts());
            testCase.addTeardown(@() close(fig));
            ax = findobj(fig, 'Type', 'axes');
            testCase.verifyEqual(numel(ax), 4);
            for k = 1:numel(ax)
                xl = findobj(ax(k), 'Type', 'ConstantLine');
                testCase.verifyGreaterThanOrEqual(numel(xl), 1, ...
                    sprintf('panel %d missing impact xline', k));
            end
        end

        function test_rejects_target_missing_clubhead_field(testCase)
            res = struct('sim_out', struct());
            bad_target = struct('time', (0:0.01:0.1).');
            testCase.verifyError( ...
                @() plot_error_timecourse(res, bad_target), ...
                'validator:missingField');
        end

        function test_rejects_result_missing_sim_out(testCase)
            res = struct('not_sim_out', 1);
            tgt = make_target();
            testCase.verifyError( ...
                @() plot_error_timecourse(res, tgt), ...
                'validator:missingField');
        end

        function test_runs_headless_visible_off(testCase)
            [res, tgt] = make_pair();
            opts = default_viz_options();
            opts.figure_visible = 'off';
            fig = plot_error_timecourse(res, tgt, opts);
            testCase.addTeardown(@() close(fig));
            testCase.verifyEqual(get(fig, 'Visible'), 'off');
        end
    end
end

% ----- helpers -----

function [res, tgt] = make_pair()
    tgt = make_target();
    N = numel(tgt.time);
    sim = struct();
    sim.time      = tgt.time;
    sim.butt      = tgt.butt + 1e-3 * randn(N, 3);
    sim.clubhead  = tgt.clubhead + 1e-3 * randn(N, 3);
    sim.club_quat = tgt.club_quat;
    sim.tau       = 0.5 * ones(N, 3);
    sim.omega     = 0.5 * ones(N, 3);
    res = struct('sim_out', sim);
end

function tgt = make_target()
    N = 31;
    tgt = struct();
    tgt.time      = linspace(0, 0.3, N).';
    tgt.butt      = repmat([0.0, 0.0, 1.2], N, 1) + 0.001 * (1:N).' * [1 0 0];
    tgt.clubhead  = repmat([0.0, 0.0, 0.2], N, 1) + 0.002 * (1:N).' * [0 1 0];
    tgt.club_quat = repmat([1, 0, 0, 0], N, 1);
    tgt.impact_idx = uint32(round(N * 0.8));
end

function opts = headless_opts()
    opts = default_viz_options();
    opts.figure_visible = 'off';
end

function ax = local_axes_with_ylabel(axes_arr, label_text)
    ax = matlab.graphics.GraphicsPlaceholder.empty;
    for k = 1:numel(axes_arr)
        try
            ylab = get(get(axes_arr(k), 'YLabel'), 'String');
        catch
            ylab = '';
        end
        if contains(string(ylab), label_text)
            ax = axes_arr(k);
            return;
        end
    end
    error('local_axes_with_ylabel:notFound', ...
        'No axis with ylabel containing "%s".', label_text);
end
