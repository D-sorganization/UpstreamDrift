function objective = Maximum_CHSz(data)
    % MAXIMUM_CHSZ Computes the maximum Clubhead Speed z-component.
    %   objective = Maximum_CHSz(data) finds the maximum value of
    %   data.Nominal.SigCHvz and scales it by 2.23694 (conversion factor).
    %
    %   Input:
    %       data - Structure containing simulation results, specifically
    %              data.Nominal.SigCHvz.
    %
    %   Output:
    %       objective - The calculated maximum value.

    arguments
        data struct
    end

    objective = max(data.Nominal.SigCHvz) * 2.23694;
end
