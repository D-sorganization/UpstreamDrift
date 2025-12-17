function objective = Minimum_CHSz(data)
    % MINIMUM_CHSZ Computes the minimum Clubhead Speed z-component.
    %   objective = Minimum_CHSz(data) finds the minimum value of
    %   data.Nominal.SigCHvz and scales it by 2.23694 (conversion factor).
    %
    %   Input:
    %       data - Structure containing simulation results, specifically
    %              data.Nominal.SigCHvz.
    %
    %   Output:
    %       objective - The calculated minimum value.

    arguments
        data struct
    end

    objective = min(data.Nominal.SigCHvz) * 2.23694;
end
