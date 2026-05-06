classdef OptimizationProgressDashboard < handle
%OPTIMIZATIONPROGRESSDASHBOARD  Live-updating dashboard for Option 1 solvers.
%
%   dash = OptimizationProgressDashboard(target, opts) creates a four-panel
%   figure (cost, |grad J|, step size, theta-on-bounds) and starts a timer
%   that refreshes the panels at opts.refresh_hz Hz (default and cap = 5).
%
%   stop_fcn = dash.outputFcn();
%   options.OutputFcn = stop_fcn;            % fmincon / surrogateopt / MultiStart
%
%   The OutputFcn is the only "hot path" callback hit by the optimizer.
%   It pushes (x, optimValues, state) into a polling queue (or local cell
%   buffer fallback) and returns immediately. Drawing happens off the
%   optimizer thread on the timer tick.
%
%   dash.close() stops the timer, releases the queue, and deletes the
%   figure. It is idempotent.
%
%   GitHub issue: #027 / #3996.
%
%   See also: dashboard_outputfcn, MultiStartParallelCoords.

    properties (SetAccess = private)
        Figure          matlab.ui.Figure   = matlab.ui.Figure.empty
        AxCost          = []
        AxGrad          = []
        AxStep          = []
        AxTheta         = []
        Target          struct
        Options         struct
        Queue           = []   % parallel.pool.PollableDataQueue or []
        FallbackBuffer  cell   = {}
        RefreshTimer    = []
        IterationCount  double = 0
        RedrawCount     double = 0
        History         struct
        BestTheta       double = []
        BestCost        double = Inf
        Closed          logical = false
    end

    methods
        function obj = OptimizationProgressDashboard(target, opts)
            arguments
                target (1,1) struct
                opts   (1,1) struct = OptimizationProgressDashboard.default_options()
            end
            % Merge defaults so callers can override only some fields.
            opts = OptimizationProgressDashboard.merge_defaults(opts);

            % --- Preconditions (DbC) -----------------------------------
            if opts.refresh_hz <= 0
                error("OptimizationProgressDashboard:badRefresh", ...
                    "opts.refresh_hz must be > 0 (got %g)", opts.refresh_hz);
            end
            if opts.refresh_hz > 5
                opts.refresh_hz = 5;  % shared/VISUALIZATION_SPEC.md cap
            end
            if ~(isnumeric(opts.history_limit) && opts.history_limit > 0)
                error("OptimizationProgressDashboard:badHistoryLimit", ...
                    "opts.history_limit must be a positive number");
            end

            obj.Target  = target;
            obj.Options = opts;
            obj.History = obj.empty_history();

            % --- Build figure + 4 axes ---------------------------------
            visibility = 'on';
            if isfield(opts, 'visible') && ~opts.visible
                visibility = 'off';
            end
            obj.Figure = figure( ...
                'Visible', visibility, ...
                'Name',    'Optimization Progress', ...
                'NumberTitle', 'off', ...
                'Color',   'w', ...
                'Position', [100 100 900 600]);
            tlo = tiledlayout(obj.Figure, 2, 2, ...
                'Padding', 'compact', 'TileSpacing', 'compact');
            obj.AxCost  = nexttile(tlo); title(obj.AxCost,  'Cost vs iteration');
            obj.AxGrad  = nexttile(tlo); title(obj.AxGrad,  '|grad J|');
            obj.AxStep  = nexttile(tlo); title(obj.AxStep,  'Step size');
            obj.AxTheta = nexttile(tlo); title(obj.AxTheta, 'theta on bounds');
            xlabel(obj.AxCost, 'iteration'); ylabel(obj.AxCost, 'fval');
            xlabel(obj.AxGrad, 'iteration'); ylabel(obj.AxGrad, '|grad J|');
            xlabel(obj.AxStep, 'iteration'); ylabel(obj.AxStep, 'step');

            % --- Build queue (best-effort) -----------------------------
            try
                obj.Queue = parallel.pool.PollableDataQueue;
            catch
                obj.Queue = [];   % fallback: direct cell buffer
            end

            % --- Start timer -------------------------------------------
            period = max(1 / opts.refresh_hz, 0.05);
            obj.RefreshTimer = timer( ...
                'ExecutionMode', 'fixedSpacing', ...
                'Period',        round(period * 1000) / 1000, ...
                'BusyMode',      'drop', ...
                'Name',          'OptProgressDashboardTimer', ...
                'TimerFcn',      @(~,~) obj.onTimerTick());
            if isfield(opts, 'autostart_timer') && ~opts.autostart_timer
                % Tests construct with autostart_timer=false to avoid races.
            else
                start(obj.RefreshTimer);
            end
        end

        function fcn = outputFcn(obj)
            %OUTPUTFCN  Return a function handle for OutputFcn options.
            fcn = @(x, optimValues, state) ...
                dashboard_outputfcn(obj, x, optimValues, state);
        end

        function pushIteration(obj, x, optimValues, state)
            %PUSHITERATION  O(1) push from the optimizer thread.
            rec = struct( ...
                'x',      x(:), ...
                'fval',   local_get(optimValues, 'fval', NaN), ...
                'iter',   local_get(optimValues, 'iteration', obj.IterationCount + 1), ...
                'grad',   local_get(optimValues, 'firstorderopt', NaN), ...
                'step',   local_get(optimValues, 'stepsize', NaN), ...
                'state',  string(state));
            if ~isempty(obj.Queue)
                try
                    send(obj.Queue, rec);
                    obj.IterationCount = obj.IterationCount + 1;
                    return;
                catch
                    % fall through to buffer
                end
            end
            obj.FallbackBuffer{end+1} = rec; %#ok<AGROW>
            obj.IterationCount = obj.IterationCount + 1;
        end

        function drainAndRender(obj)
            %DRAINANDRENDER  Pull queued records, update history, redraw.
            obj.drainQueue();
            obj.renderPanels();
            obj.RedrawCount = obj.RedrawCount + 1;
        end

        function close(obj)
            %CLOSE  Idempotent teardown.
            if obj.Closed
                return;
            end
            obj.Closed = true;
            try
                if ~isempty(obj.RefreshTimer) && isvalid(obj.RefreshTimer)
                    stop(obj.RefreshTimer);
                    delete(obj.RefreshTimer);
                end
            catch
            end
            obj.RefreshTimer = [];
            obj.Queue = [];
            obj.FallbackBuffer = {};
            try
                if ~isempty(obj.Figure) && isvalid(obj.Figure)
                    delete(obj.Figure);
                end
            catch
            end
        end

        function delete(obj)
            obj.close();
        end
    end

    methods (Access = private)
        function onTimerTick(obj)
            try
                obj.drainAndRender();
            catch
                % Never let a timer callback throw and orphan the timer.
            end
        end

        function drainQueue(obj)
            % Pull from queue
            if ~isempty(obj.Queue)
                while true
                    [rec, ok] = poll(obj.Queue, 0);
                    if ~ok, break; end
                    obj.appendRecord(rec);
                end
            end
            % Drain fallback buffer
            if ~isempty(obj.FallbackBuffer)
                buf = obj.FallbackBuffer;
                obj.FallbackBuffer = {};
                for k = 1:numel(buf)
                    obj.appendRecord(buf{k});
                end
            end
        end

        function appendRecord(obj, rec)
            obj.History.iter(end+1, 1) = double(rec.iter);
            obj.History.fval(end+1, 1) = double(rec.fval);
            obj.History.grad(end+1, 1) = double(rec.grad);
            obj.History.step(end+1, 1) = double(rec.step);
            % Cap memory growth
            cap = obj.Options.history_limit;
            n = numel(obj.History.iter);
            if n > cap
                drop = n - cap;
                obj.History.iter(1:drop) = [];
                obj.History.fval(1:drop) = [];
                obj.History.grad(1:drop) = [];
                obj.History.step(1:drop) = [];
            end
            if isfinite(rec.fval) && rec.fval < obj.BestCost
                obj.BestCost  = double(rec.fval);
                obj.BestTheta = double(rec.x);
            end
        end

        function renderPanels(obj)
            if isempty(obj.Figure) || ~isvalid(obj.Figure), return; end
            iters = obj.History.iter;
            if isempty(iters), return; end
            % Cost
            f = obj.History.fval;
            f(f <= 0 | ~isfinite(f)) = NaN;
            semilogy(obj.AxCost, iters, f, 'b-o', 'MarkerSize', 3);
            title(obj.AxCost, sprintf('Cost (iter=%d)', iters(end)));
            grid(obj.AxCost, 'on');
            % Grad
            g = obj.History.grad;
            g(g <= 0 | ~isfinite(g)) = NaN;
            semilogy(obj.AxGrad, iters, g, 'r-');
            grid(obj.AxGrad, 'on');
            % Step
            s = obj.History.step;
            s(s <= 0 | ~isfinite(s)) = NaN;
            semilogy(obj.AxStep, iters, s, 'g-');
            grid(obj.AxStep, 'on');
            % Theta-on-bounds
            if ~isempty(obj.BestTheta)
                d = numel(obj.BestTheta);
                cla(obj.AxTheta);
                barh(obj.AxTheta, 1:d, obj.BestTheta);
                ylabel(obj.AxTheta, 'coef index');
                title(obj.AxTheta, 'best theta');
            end
            drawnow('limitrate');
        end
    end

    methods (Static)
        function opts = default_options()
            opts = struct();
            opts.refresh_hz                = 5;
            opts.show_thumbnail_trajectory = true;
            opts.history_limit             = 1000;
            opts.visible                   = true;
            opts.autostart_timer           = true;
        end

        function out = merge_defaults(opts)
            d = OptimizationProgressDashboard.default_options();
            fns = fieldnames(d);
            out = opts;
            for k = 1:numel(fns)
                if ~isfield(out, fns{k})
                    out.(fns{k}) = d.(fns{k});
                end
            end
        end

        function h = empty_history()
            h = struct('iter', [], 'fval', [], 'grad', [], 'step', []);
        end
    end
end

% =====================================================================
function v = local_get(s, name, default)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = double(s.(name));
    else
        v = double(default);
    end
end
