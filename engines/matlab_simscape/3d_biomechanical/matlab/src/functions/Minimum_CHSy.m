function objective = Minimum_CHSy(data)
    % MINIMUM_CHSY Computes the minimum Clubhead Speed y-component.
    %   objective = Minimum_CHSy(data) finds the minimum value of
    %   data.Nominal.SigCHvy and scales it by 2.23694 (conversion factor).
    %
    %   Input:
    %       data - Structure containing simulation results, specifically
    %              data.Nominal.SigCHvy.
    %
    %   Output:
    %       objective - The calculated minimum value.

    arguments
        data struct
    end

    objective = min(data.Nominal.SigCHvy) * 2.23694;
end
