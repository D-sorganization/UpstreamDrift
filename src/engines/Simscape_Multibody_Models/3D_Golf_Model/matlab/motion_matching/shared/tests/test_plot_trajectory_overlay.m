classdef test_plot_trajectory_overlay < matlab.unittest.TestCase
%TEST_PLOT_TRAJECTORY_OVERLAY  Headless tests for View 1 still image.
%
%   GitHub issue: #3989.

    methods (TestClassSetup)
        function add_paths(testCase)
            here = fileparts(mfilename('fullpath'));
            shared = fileparts(here);
            addpath(shared);
            testCase.addTeardown(@() rmpath(shared));
        end
    end

    methods (Test)
        function test_default_viz_options_contract(testCase)
            opts = default_viz_options();
            required = ["color_measured", "color_simulated", "color_error", ...
                        "dpi", "fontsize_axes", "fontsize_title", ...
                        "video_fps", "video_format", "show_impact_marker", ...
                        "tight_inset", "visible", "save_path", ...
                        "live_refresh_hz", "trace_alpha"];
            for f = required
                testCase.verifyTrue(isfield(opts, f), ...
                    sprintf("default_viz_options missing field %s", f));
            end
            testCase.verifyEqual(opts.color_measured,  "#1f77b4");
            testCase.verifyEqual(opts.color_simulated, "#d62728");
            testCase.verifyEqual(opts.color_error,     "#7f7f7f");
            testCase.verifyTrue(isnumeric(opts.dpi));
            testCase.verifyTrue(isnumeric(opts.video_fps));
            testCase.verifyTrue(islogical(opts.show_impact_marker));
        end

        function test_plot_returns_figure_with_two_3d_axes(testCase)
            [result, target, opts] = make_inputs();
            fig = plot_trajectory_overlay(result, target, opts);
            cleanup = onCleanup(@() close(fig));
            testCase.verifyTrue(isgraphics(fig, 'figure'));
            axs = findall(fig, 'Type', 'axes', '-not', 'Tag', 'error_inset');
            testCase.verifyGreaterThanOrEqual(numel(axs), 2);
        end

        function test_plot_axis_limits_match_across_left_right(testCase)
            [result, target, opts] = make_inputs();
            fig = plot_trajectory_overlay(result, target, opts);
            cleanup = onCleanup(@() close(fig));
            axs = findall(fig, 'Type', 'axes', '-not', 'Tag', 'error_inset');
            testCase.assertGreaterThanOrEqual(numel(axs), 2);
            [~, ord] = sort(arrayfun(@(a) a.Position(1), axs));
            ax_l = axs(ord(1));
            ax_r = axs(ord(2));
            testCase.verifyEqual(ax_l.XLim, ax_r.XLim, 'AbsTol', 1e-9);
            testCase.verifyEqual(ax_l.YLim, ax_r.YLim, 'AbsTol', 1e-9);
            testCase.verifyEqual(ax_l.ZLim, ax_r.ZLim, 'AbsTol', 1e-9);
        end

        function test_plot_does_not_modify_input_structs(testCase)
            [result, target, opts] = make_inputs();
            r0 = result;
            t0 = target;
            fig = plot_trajectory_overlay(result, target, opts);
            cleanup = onCleanup(@() close(fig));
            testCase.verifyEqual(result, r0);
            testCase.verifyEqual(target, t0);
        end

        function test_input_validation_rejects_missing_target_fields(testCase)
            [result, target, opts] = make_inputs();
            bad_target = rmfield(target, "butt");
            testCase.verifyError( ...
                @() plot_trajectory_overlay(result, bad_target, opts), ...
                "validator:missingField");
        end

        function test_plot_renders_for_option2_style_struct(testCase)
            % Option-2 (surrogate) style: result includes additional
            % fields but the contract on sim_out is identical.
            [result, target, opts] = make_inputs();
            result.surrogate_loss = 0.123;
            result.coefficients   = randn(7 * 6, 1);
            fig = plot_trajectory_overlay(result, target, opts);
            cleanup = onCleanup(@() close(fig));
            testCase.verifyTrue(isgraphics(fig, 'figure'));
        end
    end
end

% =====================================================================
% Local fixture: a synthetic, deterministic minimal valid (result,target).
% =====================================================================
function [result, target, opts] = make_inputs()
    N = 60;
    t = linspace(0, 0.3, N).';
    % A simple curving path so axis limits are non-degenerate.
    head = [0.5 * sin(2 * pi * t / 0.3), ...
            0.5 * cos(2 * pi * t / 0.3), ...
            0.2 + 0.05 * t];
    butt = head - repmat([0 0 1.0], N, 1);  % 1 m shaft along -z
    quat = repmat([1 0 0 0], N, 1);

    target = struct( ...
        "time", t, ...
        "butt", butt, ...
        "clubhead", head, ...
        "club_quat", quat, ...
        "impact_idx", uint32(round(N / 2)), ...
        "source", struct("filename", "synth.xlsx", "format", "xlsx", ...
                         "subject_id", "synth", "trial_id", "1", ...
                         "sha256", repmat('0', 1, 64)));

    sim_out = struct( ...
        "time", t, ...
        "r_butt", butt + 0.005 * randn(N, 3), ...
        "r_clubhead", head + 0.005 * randn(N, 3), ...
        "q_club", quat, ...
        "solver_status", "success");

    result = struct("sim_out", sim_out, "coefficients", zeros(7 * 6, 1));
    opts = default_viz_options();
    opts.visible = "off";
end
