classdef test_compute_pointwise_errors < matlab.unittest.TestCase
%TEST_COMPUTE_POINTWISE_ERRORS  Unit tests for the pointwise-error helpers
%   that live in shared/private/ and are exercised through
%   plot_error_timecourse.  Because they are in a private/ folder we test
%   them indirectly via the public plot function in test_plot_error_timecourse,
%   plus the structural cases below using a small wrapper script.

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            shared = fullfile(here, '..');
            addpath(shared);
            testCase.addTeardown(@() rmpath(shared));
        end
    end

    methods (Test)
        function test_position_error_zero_when_sim_equals_meas(testCase)
            tgt = make_target();
            res = make_result_matching(tgt);
            fig = plot_error_timecourse(res, tgt, headless_opts());
            testCase.addTeardown(@() close(fig));
            ax = findobj(fig, 'Type', 'axes');
            % Panel 1 is the topmost subplot (last in subplot creation order
            % depends on findobj; we filter by ylabel string instead).
            ax_pos = local_axes_with_ylabel(ax, 'Position error (mm)');
            lines = findobj(ax_pos, 'Type', 'line');
            for L = lines.'
                yd = get(L, 'YData');
                testCase.verifyLessThan(max(abs(yd)), 1e-9);
            end
        end

        function test_orientation_error_zero_when_quat_equals_quat(testCase)
            tgt = make_target();
            res = make_result_matching(tgt);
            fig = plot_error_timecourse(res, tgt, headless_opts());
            testCase.addTeardown(@() close(fig));
            ax = local_axes_with_ylabel( ...
                findobj(fig, 'Type', 'axes'), 'Orientation error (deg)');
            line_h = findobj(ax, 'Type', 'line');
            yd = get(line_h(1), 'YData');
            testCase.verifyLessThan(max(abs(yd)), 1e-9);
        end

        function test_orientation_error_zero_when_quat_equals_neg_quat(testCase)
            % q and -q represent the same rotation; geodesic distance == 0.
            tgt = make_target();
            res = make_result_matching(tgt);
            res.sim_out.club_quat = -res.sim_out.club_quat;
            fig = plot_error_timecourse(res, tgt, headless_opts());
            testCase.addTeardown(@() close(fig));
            ax = local_axes_with_ylabel( ...
                findobj(fig, 'Type', 'axes'), 'Orientation error (deg)');
            line_h = findobj(ax, 'Type', 'line');
            yd = get(line_h(1), 'YData');
            testCase.verifyLessThan(max(abs(yd)), 1e-9);
        end

        function test_clubhead_speed_units_are_mph(testCase)
            % Build a synthetic 100 mph linear motion: 100 mph = 44.704 m/s.
            v_mph = 100;
            v_mps = v_mph / 2.2369362920544;
            N = 21;
            tgt = make_target_with_linear_motion(N, v_mps);
            res = make_result_matching(tgt);
            fig = plot_error_timecourse(res, tgt, headless_opts());
            testCase.addTeardown(@() close(fig));
            ax = local_axes_with_ylabel( ...
                findobj(fig, 'Type', 'axes'), 'Clubhead speed (mph)');
            lines = findobj(ax, 'Type', 'line');
            % Use interior samples (gradient endpoints are one-sided)
            for L = lines.'
                yd = get(L, 'YData');
                interior = yd(3:end-2);
                testCase.verifyEqual(mean(interior), v_mph, 'AbsTol', 1.0);
            end
        end
    end
end

% ----- shared helpers -----

function tgt = make_target()
    N = 31;
    tgt = struct();
    tgt.time      = linspace(0, 0.3, N).';
    tgt.butt      = repmat([0.0, 0.0, 1.2], N, 1) + 0.001 * (1:N).' * [1 0 0];
    tgt.clubhead  = repmat([0.0, 0.0, 0.2], N, 1) + 0.002 * (1:N).' * [0 1 0];
    tgt.club_quat = repmat([1, 0, 0, 0], N, 1);
    tgt.impact_idx = uint32(round(N * 0.8));
end

function tgt = make_target_with_linear_motion(N, v_mps)
    tgt = struct();
    tgt.time      = linspace(0, 0.1, N).';
    tgt.butt      = zeros(N, 3);
    tgt.clubhead  = [v_mps * tgt.time, zeros(N, 1), zeros(N, 1)];
    tgt.club_quat = repmat([1, 0, 0, 0], N, 1);
    tgt.impact_idx = uint32(round(N * 0.5));
end

function res = make_result_matching(tgt)
    N = numel(tgt.time);
    sim = struct();
    sim.time      = tgt.time;
    sim.butt      = tgt.butt;
    sim.clubhead  = tgt.clubhead;
    sim.club_quat = tgt.club_quat;
    sim.tau       = ones(N, 3);
    sim.omega     = ones(N, 3) * 0.5;
    res = struct('sim_out', sim);
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
