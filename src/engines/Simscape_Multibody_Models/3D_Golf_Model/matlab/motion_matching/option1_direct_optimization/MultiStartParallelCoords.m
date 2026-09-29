classdef MultiStartParallelCoords < handle
%MULTISTARTPARALLELCOORDS  Post-fit parallel-coordinate plot for MultiStart.
%
%   fig = MultiStartParallelCoords.plot(result, opts)
%
%   - X-axis: coefficient index 1..d (one tick per coefficient).
%   - Y-axis: coefficient value normalized to [0,1] per (lb,ub).
%   - One polyline per starting point (N total).
%   - Polyline colour mapped from final cost (parula colormap).
%   - The lowest-cost line is drawn last, in heavy red, on top.
%
%   GitHub issue: #027 / #3996.
%
%   See also: OptimizationProgressDashboard.

    methods (Static)
        function fig = plot(result, opts)
            arguments
                result (1,1) struct
                opts   (1,1) struct = MultiStartParallelCoords.default_options()
            end
            % --- Preconditions ----------------------------------------
            if ~isfield(result, 'start_points') || isempty(result.start_points)
                error("MultiStartParallelCoords:noStartPoints", ...
                    "result.start_points must be a non-empty d x N matrix");
            end
            if ~isfield(result, 'start_costs') || isempty(result.start_costs)
                error("MultiStartParallelCoords:noStartCosts", ...
                    "result.start_costs must be a 1 x N vector");
            end
            opts = MultiStartParallelCoords.merge_defaults(opts);

            S      = double(result.start_points);
            costs  = double(result.start_costs(:).');
            [d, N] = size(S);
            if numel(costs) ~= N
                error("MultiStartParallelCoords:sizeMismatch", ...
                    "start_costs length %d != N=%d starts", numel(costs), N);
            end

            % --- Bounds for normalization ------------------------------
            if isfield(result, 'options') && isstruct(result.options) && ...
                    isfield(result.options, 'sim') && ...
                    isfield(result.options.sim, 'joint_names') && ...
                    ~isempty(result.options.sim.joint_names)
                n_joints = numel(string(result.options.sim.joint_names));
            else
                n_joints = max(1, round(d / 7));
            end
            lb = -ones(d, 1);
            ub =  ones(d, 1);
            try
                [lb_b, ub_b] = build_coefficient_bounds(n_joints);
                if numel(lb_b) == d
                    lb = lb_b(:); ub = ub_b(:);
                end
            catch
                lb = min(S, [], 2); ub = max(S, [], 2);
                rng_w = max(ub - lb, eps);
                lb = lb - 0.05 * rng_w; ub = ub + 0.05 * rng_w;
            end
            denom = max(ub - lb, eps);
            Snorm = (S - lb) ./ denom;

            % --- Build figure -----------------------------------------
            visibility = 'on';
            if isfield(opts, 'visible') && ~opts.visible
                visibility = 'off';
            end
            fig = figure('Visible', visibility, 'Color', 'w', ...
                'Name', 'MultiStart Parallel Coords', 'NumberTitle', 'off');
            ax = axes('Parent', fig); hold(ax, 'on');

            % Colour map by cost
            finite_costs = costs(isfinite(costs));
            if isempty(finite_costs)
                cmin = 0; cmax = 1;
            else
                cmin = min(finite_costs); cmax = max(finite_costs);
                if cmax <= cmin, cmax = cmin + 1; end
            end
            cmap = parula(256);

            % --- Draw one polyline per start ---------------------------
            x = 1:d;
            for k = 1:N
                if isfinite(costs(k))
                    t = (costs(k) - cmin) / (cmax - cmin);
                else
                    t = 1;
                end
                idx = max(1, min(256, round(t * 255) + 1));
                col = cmap(idx, :);
                plot(ax, x, Snorm(:, k), '-', 'Color', col, ...
                    'LineWidth', 0.8, 'Tag', sprintf('start_%d', k));
            end
            % Best line on top
            [~, kbest] = min(costs);
            plot(ax, x, Snorm(:, kbest), 'r-', 'LineWidth', 2.5, ...
                'Tag', 'best_start');

            xlabel(ax, 'coefficient index');
            ylabel(ax, 'normalized value (0=lb, 1=ub)');
            title(ax, sprintf('MultiStart parallel coords (N=%d, d=%d)', N, d));
            grid(ax, 'on');
            ylim(ax, [-0.05, 1.05]);
            xlim(ax, [0.5, d + 0.5]);
            colormap(ax, cmap);
            cb = colorbar(ax);
            cb.Label.String = 'final cost';
            caxis(ax, [cmin, cmax]);
        end

        function opts = default_options()
            opts = struct();
            opts.visible = true;
        end

        function out = merge_defaults(opts)
            d = MultiStartParallelCoords.default_options();
            fns = fieldnames(d);
            out = opts;
            for k = 1:numel(fns)
                if ~isfield(out, fns{k})
                    out.(fns{k}) = d.(fns{k});
                end
            end
        end
    end
end
