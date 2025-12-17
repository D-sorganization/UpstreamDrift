function [BASEQ_updated, ZTCFQ_updated, DELTAQ_updated] = calculateTotalWorkAndPower3D(BASEQ, ZTCFQ, DELTAQ)
% CALCULATETOTALWORKANDPOWER3D Computes total and fractional work and power for segments.
%   [BASEQ_updated, ZTCFQ_updated, DELTAQ_updated] = CALCULATETOTALWORKANDPOWER3D(BASEQ, ZTCFQ, DELTAQ)
%   Calculates the total work (Linear Work + Angular Work) and total power
%   (Linear Power + Angular Power) for various body segments in the BASEQ,
%   ZTCFQ, and DELTAQ tables. It also calculates the fractional work and
%   power contributed by the ZTCF and DELTA components relative to the BASE
%   case and adds these as new columns to the BASEQ table.
%
%   This function is the converted version of SCRIPT_TotalWorkandPowerCalculation_3D.m.
%   It operates on tables expected to be on a uniform time grid (like the 'Q' tables),
%   and assumes that linear and angular work/power columns have already been added
%   (e.g., by calculateWorkAndImpulse3D).
%
%   Input:
%       BASEQ  - Table containing Base data on a uniform time grid,
%                including linear/angular work and power columns.
%       ZTCFQ  - Table containing ZTCF data on the same uniform time grid,
%                including linear/angular work and power columns.
%       DELTAQ - Table containing Delta data on the same uniform time grid,
%                including linear/angular work and power columns.
%
%   Output:
%       BASEQ_updated  - Updated BASEQ table with total and fractional work/power columns.
%       ZTCFQ_updated  - Updated ZTCFQ table with total work/power columns.
%       DELTAQ_updated - Updated DELTAQ table with total work/power columns.

% Assign input tables to internal variables for modification
BASEQ_updated = BASEQ;
ZTCFQ_updated = ZTCFQ;
DELTAQ_updated = DELTAQ;

% --- Generate Total Work and Power Vectors ---
% Sum the linear and angular components for each segment and table.
% Ensure the required linear and angular work/power columns exist before summing.

% Define the required columns for calculating total work and power
requiredWorkPowerInputCols = {
    'LSLinearWorkonArm', 'LSAngularWorkonArm', 'LSonArmLinearPower', 'LSonArmAngularPower', ...
    'RSLinearWorkonArm', 'RSAngularWorkonArm', 'RSonArmLinearPower', 'RSonArmAngularPower', ...
    'LELinearWorkonForearm', 'LEAngularWorkonForearm', 'LEonForearmLinearPower', 'LEonForearmAngularPower', ...
    'RELinearWorkonForearm', 'REAngularWorkonForearm', 'REonForearmLinearPower', 'REonForearmAngularPower', ...
    'LHLinearWorkonClub', 'LHAngularWorkonClub', 'LHonClubLinearPower', 'LHonClubAngularPower', ... % Assuming LH/RH naming
    'RHLinearWorkonClub', 'RHAngularWorkonClub', 'RHonClubLinearPower', 'RHonClubAngularPower' % Assuming LH/RH naming
};

% Helper function to check if all columns exist in a table
checkCols = @(tbl, cols) all(ismember(cols, tbl.Properties.VariableNames));

% Check if input tables have the necessary columns
if checkCols(BASEQ_updated, requiredWorkPowerInputCols) && ...
   checkCols(ZTCFQ_updated, requiredWorkPowerInputCols) && ...
   checkCols(DELTAQ_updated, requiredWorkPowerInputCols)

    % BASEQ
    BASEQ_updated.TotalLSWork = BASEQ_updated.LSLinearWorkonArm + BASEQ_updated.LSAngularWorkonArm;
    BASEQ_updated.TotalRSWork = BASEQ_updated.RSLinearWorkonArm + BASEQ_updated.RSAngularWorkonArm;
    BASEQ_updated.TotalLEWork = BASEQ_updated.LELinearWorkonForearm + BASEQ_updated.LEAngularWorkonForearm;
    BASEQ_updated.TotalREWork = BASEQ_updated.RELinearWorkonForearm + BASEQ_updated.REAngularWorkonForearm;
    BASEQ_updated.TotalLHWork = BASEQ_updated.LHLinearWorkonClub + BASEQ_updated.LHAngularWorkonClub; % Assuming LH/RH naming
    BASEQ_updated.TotalRHWork = BASEQ_updated.RHLinearWorkonClub + BASEQ_updated.RHAngularWorkonClub; % Assuming LH/RH naming

    BASEQ_updated.TotalLSPower = BASEQ_updated.LSonArmLinearPower + BASEQ_updated.LSonArmAngularPower;
    BASEQ_updated.TotalRSPower = BASEQ_updated.RSonArmLinearPower + BASEQ_updated.RSonArmAngularPower;
    BASEQ_updated.TotalLEPower = BASEQ_updated.LEonForearmLinearPower + BASEQ_updated.LEonForearmAngularPower;
    BASEQ_updated.TotalREPower = BASEQ_updated.REonForearmLinearPower + BASEQ_updated.REonForearmAngularPower;
    BASEQ_updated.TotalLHPower = BASEQ_updated.LHonClubLinearPower + BASEQ_updated.LHonClubAngularPower; % Assuming LH/RH naming
    BASEQ_updated.TotalRHPower = BASEQ_updated.RHonClubLinearPower + BASEQ_updated.RHonClubAngularPower; % Assuming LH/RH naming

    % ZTCFQ
    ZTCFQ_updated.TotalLSWork = ZTCFQ_updated.LSLinearWorkonArm + ZTCFQ_updated.LSAngularWorkonArm;
    ZTCFQ_updated.TotalRSWork = ZTCFQ_updated.RSLinearWorkonArm + ZTCFQ_updated.RSAngularWorkonArm;
    ZTCFQ_updated.TotalLEWork = ZTCFQ_updated.LELinearWorkonForearm + ZTCFQ_updated.LEAngularWorkonForearm;
    ZTCFQ_updated.TotalREWork = ZTCFQ_updated.RELinearWorkonForearm + ZTCFQ_updated.REAngularWorkonForearm;
    ZTCFQ_updated.TotalLHWork = ZTCFQ_updated.LHLinearWorkonClub + ZTCFQ_updated.LHAngularWorkonClub; % Assuming LH/RH naming
    ZTCFQ_updated.TotalRHWork = ZTCFQ_updated.RHLinearWorkonClub + ZTCFQ_updated.RHAngularWorkonClub; % Assuming LH/RH naming

    ZTCFQ_updated.TotalLSPower = ZTCFQ_updated.LSonArmLinearPower + ZTCFQ_updated.LSonArmAngularPower;
    ZTCFQ_updated.TotalRSPower = ZTCFQ_updated.RSonArmLinearPower + ZTCFQ_updated.RSonArmAngularPower;
    ZTCFQ_updated.TotalLEPower = ZTCFQ_updated.LEonForearmLinearPower + ZTCFQ_updated.LEonForearmAngularPower;
    ZTCFQ_updated.TotalREPower = ZTCFQ_updated.REonForearmLinearPower + ZTCFQ_updated.REonForearmAngularPower;
    ZTCFQ_updated.TotalLHPower = ZTCFQ_updated.LHonClubLinearPower + ZTCFQ_updated.LHonClubAngularPower; % Assuming LH/RH naming
    ZTCFQ_updated.TotalRHPower = ZTCFQ_updated.RHonClubLinearPower + ZTCFQ_updated.RHonClubAngularPower; % Assuming LH/RH naming

    % DELTAQ
    DELTAQ_updated.TotalLSWork = DELTAQ_updated.LSLinearWorkonArm + DELTAQ_updated.LSAngularWorkonArm;
    DELTAQ_updated.TotalRSWork = DELTAQ_updated.RSLinearWorkonArm + DELTAQ_updated.RSAngularWorkonArm;
    DELTAQ_updated.TotalLEWork = DELTAQ_updated.LELinearWorkonForearm + DELTAQ_updated.LEAngularWorkonForearm;
    DELTAQ_updated.TotalREWork = DELTAQ_updated.RELinearWorkonForearm + DELTAQ_updated.REAngularWorkonForearm;
    DELTAQ_updated.TotalLHWork = DELTAQ_updated.LHLinearWorkonClub + DELTAQ_updated.LHAngularWorkonClub; % Assuming LH/RH naming
    DELTAQ_updated.TotalRHWork = DELTAQ_updated.RHLinearWorkonClub + DELTAQ_updated.RHAngularWorkonClub; % Assuming LH/RH naming

    DELTAQ_updated.TotalLSPower = DELTAQ_updated.LSonArmLinearPower + DELTAQ_updated.LSonArmAngularPower;
    DELTAQ_updated.TotalRSPower = DELTAQ_updated.RSonArmLinearPower + DELTAQ_updated.RSonArmAngularPower;
    DELTAQ_updated.TotalLEPower = DELTAQ_updated.LEonForearmLinearPower + DELTAQ_updated.LEonForearmAngularPower;
    DELTAQ_updated.TotalREPower = DELTAQ_updated.REonForearmLinearPower + DELTAQ_updated.REonForearmAngularPower;
    DELTAQ_updated.TotalLHPower = DELTAQ_updated.LHonClubLinearPower + DELTAQ_updated.LHonClubAngularPower; % Assuming LH/RH naming
    DELTAQ_updated.TotalRHPower = DELTAQ_updated.RHonClubLinearPower + DELTAQ_updated.RHonClubAngularPower; % Assuming LH/RH naming

else
    warning('Required linear/angular work/power columns not found in input tables. Cannot calculate total work/power.');
    % Return updated tables as is if columns are missing
    BASEQ_updated = BASEQ;
    ZTCFQ_updated = ZTCFQ;
    DELTAQ_updated = DELTAQ;
    return; % Exit the function
end


% --- Calculate the fraction of work and power being done by the ZTCF and DELTA ---
% This calculation is added as new columns to the BASEQ table.
% Avoid division by zero or very small numbers in the denominator (BASEQ Total values).
% Replace Inf/NaN results with 0 or NaN as appropriate.

% Define the required Total columns for fractional calculations
requiredTotalCols = {
    'TotalLSWork', 'TotalLSPower', 'TotalRSWork', 'TotalRSPower', ...
    'TotalLEWork', 'TotalLEPower', 'TotalREWork', 'TotalREPower', ...
    'TotalLHWork', 'TotalLHPower', 'TotalRHWork', 'TotalRHPower' % Assuming LH/RH naming
};

% Check if input tables have the necessary total columns
if checkCols(BASEQ_updated, requiredTotalCols) && ...
   checkCols(ZTCFQ_updated, requiredTotalCols) && ...
   checkCols(DELTAQ_updated, requiredTotalCols)

    % Helper function to safely divide, handling division by zero or near-zero denominators
    safeDivide = @(numerator, denominator) numerator ./ (denominator + eps(class(denominator)) * (abs(denominator) < eps(class(denominator)))); % Use eps based on data type

    % Calculate Fraction of Work and Power Done by ZTCF
    BASEQ_updated.ZTCFQLSFractionalWork = safeDivide(ZTCFQ_updated.TotalLSWork, BASEQ_updated.TotalLSWork);
    BASEQ_updated.ZTCFQRSFractionalWork = safeDivide(ZTCFQ_updated.TotalRSWork, BASEQ_updated.TotalRSWork);
    BASEQ_updated.ZTCFQLEFractionalWork = safeDivide(ZTCFQ_updated.TotalLEWork, BASEQ_updated.TotalLEWork);
    BASEQ_updated.ZTCFQREFractionalWork = safeDivide(ZTCFQ_updated.TotalREWork, BASEQ_updated.TotalREWork);
    BASEQ_updated.ZTCFQLHFractionalWork = safeDivide(ZTCFQ_updated.TotalLHWork, BASEQ_updated.TotalLHWork); % Assuming LH/RH naming
    BASEQ_updated.ZTCFQRHFractionalWork = safeDivide(ZTCFQ_updated.TotalRHWork, BASEQ_updated.TotalRHWork); % Assuming LH/RH naming

    BASEQ_updated.ZTCFQLSFractionalPower = safeDivide(ZTCFQ_updated.TotalLSPower, BASEQ_updated.TotalLSPower);
    BASEQ_updated.ZTCFQRSFractionalPower = safeDivide(ZTCFQ_updated.TotalRSPower, BASEQ_updated.TotalRSPower);
    BASEQ_updated.ZTCFQLEFractionalPower = safeDivide(ZTCFQ_updated.TotalLEPower, BASEQ_updated.TotalLEPower);
    BASEQ_updated.ZTCFQREFractionalPower = safeDivide(ZTCFQ_updated.TotalREPower, BASEQ_updated.TotalREPower);
    BASEQ_updated.ZTCFQLHFractionalPower = safeDivide(ZTCFQ_updated.TotalLHPower, BASEQ_updated.TotalLHPower); % Assuming LH/RH naming
    BASEQ_updated.ZTCFQRHFractionalPower = safeDivide(ZTCFQ_updated.TotalRHPower, BASEQ_updated.TotalRHPower); % Assuming LH/RH naming

    % Calculate Fraction of Work and Power Done by DELTA
    BASEQ_updated.DELTAQLSFractionalWork = safeDivide(DELTAQ_updated.TotalLSWork, BASEQ_updated.TotalLSWork);
    BASEQ_updated.DELTAQRSFractionalWork = safeDivide(DELTAQ_updated.TotalRSWork, BASEQ_updated.TotalRSWork);
    BASEQ_updated.DELTAQLEFractionalWork = safeDivide(DELTAQ_updated.TotalLEWork, BASEQ_updated.TotalLEWork);
    BASEQ_updated.DELTAQREFractionalWork = safeDivide(DELTAQ_updated.TotalREWork, BASEQ_updated.TotalREWork);
    BASEQ_updated.DELTAQLHFractionalWork = safeDivide(DELTAQ_updated.TotalLHWork, BASEQ_updated.TotalLHWork); % Assuming LH/RH naming
    BASEQ_updated.DELTAQRHFractionalWork = safeDivide(DELTAQ_updated.TotalRHWork, BASEQ_updated.TotalRHWork); % Assuming LH/RH naming

    BASEQ_updated.DELTAQLSFractionalPower = safeDivide(DELTAQ_updated.TotalLSPower, BASEQ_updated.TotalLSPower);
    BASEQ_updated.DELTAQRSFractionalPower = safeDivide(DELTAQ_updated.TotalRSPower, BASEQ_updated.TotalRSPower);
    BASEQ_updated.DELTAQLEFractionalPower = safeDivide(DELTAQ_updated.TotalLEPower, BASEQ_updated.TotalLEPower);
    BASEQ_updated.DELTAQREFractionalPower = safeDivide(DELTAQ_updated.TotalREPower, BASEQ_updated.TotalREPower);
    BASEQ_updated.DELTAQLHFractionalPower = safeDivide(DELTAQ_updated.TotalLHPower, BASEQ_updated.TotalLHPower); % Assuming LH/RH naming
    BASEQ_updated.DELTAQRHFractionalPower = safeDivide(DELTAQ_updated.TotalRHPower, BASEQ_updated.TotalRHPower); % Assuming LH/RH naming

else
    warning('Required total work/power columns not found for fractional calculations. Skipping fractional calculations.');
    % Initialize fractional columns to NaN if needed for consistency
    fractionalWorkFields = {'ZTCFQLSFractionalWork', 'ZTCFQRSFractionalWork', 'ZTCFQLEFractionalWork', 'ZTCFQREFractionalWork', 'ZTCFQLHFractionalWork', 'ZTCFQRHFractionalWork', ...
                            'DELTAQLSFractionalWork', 'DELTAQRSFractionalWork', 'DELTAQLEFractionalWork', 'DELTAQREFractionalWork', 'DELTAQLHFractionalWork', 'DELTAQRHFractionalWork'};
    fractionalPowerFields = {'ZTCFQLSFractionalPower', 'ZTCFQRSFractionalPower', 'ZTCFQLEFractionalPower', 'ZTCFQREFractionalPower', 'ZTCFQLHFractionalPower', 'ZTCFQRHFractionalPower', ...
                             'DELTAQLSFractionalPower', 'DELTAQRSFractionalPower', 'DELTAQLEFractionalPower', 'DELTAQREFractionalPower', 'DELTAQLHFractionalPower', 'DELTAQRHFractionalPower'};

     % Pre-create NaN column for performance (avoid repeated allocation)
     % Use (:) to force independent copies for each table column
     nan_col = NaN(height(BASEQ_updated), 1);
     for k = 1:length(fractionalWorkFields)
         BASEQ_updated.(fractionalWorkFields{k}) = nan_col(:);
     end
     for k = 1:length(fractionalPowerFields)
         BASEQ_updated.(fractionalPowerFields{k}) = nan_col(:);
     end
end


% No need for explicit 'clear' statements within a function.
% All intermediate variables are local and cleared automatically.

end
