function objective = Maximum_CHSy(data)
    % MAXIMUM_CHSY Computes the maximum Clubhead Speed y-component.
    %   objective = Maximum_CHSy(data) finds the maximum value of
    %   data.Nominal.SigCHvy and scales it by 2.23694 (conversion factor).
    %
    %   Input:
    %       data - Structure containing simulation results, specifically
    %              data.Nominal.SigCHvy.
    %
    %   Output:
    %       objective - The calculated maximum value.

    arguments
        data struct
    end

    objective = max(data.Nominal.SigCHvy) * 2.23694;
end
