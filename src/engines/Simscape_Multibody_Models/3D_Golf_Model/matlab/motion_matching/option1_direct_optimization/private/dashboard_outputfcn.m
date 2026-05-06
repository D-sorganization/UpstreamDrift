function stop = dashboard_outputfcn(dashboard, x, optimValues, state)
%DASHBOARD_OUTPUTFCN  Generic OutputFcn for fmincon/MultiStart/surrogateopt.
%
%   STOP = DASHBOARD_OUTPUTFCN(DASHBOARD, X, OPTIMVALUES, STATE) pushes the
%   iteration record into DASHBOARD's queue and returns STOP=false. The
%   push is non-blocking so the optimizer is not throttled.
%
%   The signature is compatible with all three Option-1 solvers:
%     - fmincon:       OutputFcn(x, optimValues, state)
%     - MultiStart:    output function via createOptimProblem (same sig)
%     - surrogateopt:  PlotFcn / OutputFcn (same sig)
%
%   GitHub issue: #027 / #3996.
%
%   See also: OptimizationProgressDashboard.

    arguments
        dashboard
        x                                          %#ok<INUSA>
        optimValues
        state
    end

    stop = false;
    if ~isa(dashboard, 'OptimizationProgressDashboard')
        return;
    end
    if ~isstruct(optimValues)
        optimValues = struct();
    end
    try
        if isvalid(dashboard) && ~dashboard.Closed
            dashboard.pushIteration(x, optimValues, state);
        end
    catch
        % Never let a viz callback abort an optimization.
    end
end
