function objective = Maximum_CHSx(data)
    % MAXIMUM_CHSX Computes the maximum Clubhead Speed x-component.
    %   objective = Maximum_CHSx(data) finds the maximum value of
    %   data.Nominal.SigCHvx and scales it by 2.23694 (conversion factor).
    %
    %   Input:
    %       data - Structure containing simulation results, specifically
    %              data.Nominal.SigCHvx.
    %
    %   Output:
    %       objective - The calculated maximum value.

    arguments
        data struct
    end

    objective = max(data.Nominal.SigCHvx) * 2.23694;
end
