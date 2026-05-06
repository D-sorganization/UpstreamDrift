function so_opts = build_surrogateopt_options(opts, output_capture)
%BUILD_SURROGATEOPT_OPTIONS  Construct optimoptions for surrogateopt.
%
%   SO_OPTS = BUILD_SURROGATEOPT_OPTIONS(OPTS, OUTPUT_CAPTURE) returns the
%   optimoptions object passed to MATLAB's surrogateopt. Plotting is
%   disabled when running headless (no Java AWT) to avoid display errors.
%
%   Preconditions:
%     - OPTS is a 1x1 struct with at least the field
%       surrogate_max_evals (positive integer).
%     - OUTPUT_CAPTURE is a function handle implementing the surrogateopt
%       OutputFcn signature.
%
%   Postconditions:
%     - so_opts is an optimoptions object with MaxFunctionEvaluations set.
%
%   GitHub issue: #026 / #3995.
%
%   See also: FIT_SWING_SURROGATEOPT.
    arguments
        opts (1,1) struct
        output_capture (1,1) function_handle
    end

    max_evals = double(opts.surrogate_max_evals);

    use_parallel = false;
    if isfield(opts, "use_parallel")
        use_parallel = logical(opts.use_parallel);
    end

    headless = local_is_headless();
    if headless || (isfield(opts, "plot_fcn") && opts.plot_fcn == "")
        plot_fcn = [];
    else
        plot_fcn = 'optimplotbestf';
    end

    so_opts = optimoptions('surrogateopt', ...
        'MaxFunctionEvaluations', max_evals, ...
        'UseParallel',            use_parallel, ...
        'Display',                'off', ...
        'PlotFcn',                plot_fcn, ...
        'OutputFcn',              output_capture);

    if isfield(opts, "min_surrogate_points") && ~isempty(opts.min_surrogate_points)
        try
            so_opts = optimoptions(so_opts, ...
                'MinSurrogatePoints', double(opts.min_surrogate_points));
        catch
            % Older releases may name this differently; ignore.
        end
    end
    if isfield(opts, "min_sample_distance") && ~isempty(opts.min_sample_distance)
        try
            so_opts = optimoptions(so_opts, ...
                'MinSampleDistance', double(opts.min_sample_distance));
        catch
        end
    end
end

% =====================================================================
function tf = local_is_headless()
%LOCAL_IS_HEADLESS  Best-effort detection of a no-display environment.
    tf = false;
    try
        if usejava('desktop') == false || ~feature('ShowFigureWindows')
            tf = true;
        end
    catch
        tf = true;
    end
end
