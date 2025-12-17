function [SummaryTable, ClubQuiverAlphaReversal, ClubQuiverMaxCHS, ClubQuiverZTCFAlphaReversal, ClubQuiverDELTAAlphaReversal] = generateSummaryTableAndQuiverData3D(BASEQ, ZTCFQ, DELTAQ)
% GENERATESUMMARYTABLEANDQUIVERDATA3D Calculates key values and generates data for quiver plots.
%   [SummaryTable, ClubQuiverAlphaReversal, ClubQuiverMaxCHS, ...
%    ClubQuiverZTCFAlphaReversal, ClubQuiverDELTAAlphaReversal] = ...
%    GENERATESUMMARYTABLEANDQUIVERDATA3D(BASEQ, ZTCFQ, DELTAQ)
%   Analyzes the BASEQ, ZTCFQ, and DELTAQ tables (expected on a uniform
%   time grid) to find maximum speeds, times of specific events (like
%   alpha reversal and zero AoA), and generates data structures containing
%   coordinates and vectors at these key times for potential quiver plotting.
%
%   This function is the converted version of SCRIPT_TableofValues_3D.m.
%
%   Input:
%       BASEQ  - Table containing Base data on a uniform time grid.
%       ZTCFQ  - Table containing ZTCF data on the same uniform time grid.
%       DELTAQ - Table containing Delta data on the same uniform time grid.
%                These tables are expected to contain various kinematic
%                and force/torque data, including 'Time', 'CHS (mph)',
%                'Hand Speed (mph)', 'BaseAV', 'ChestAV', 'LScapAV',
%                'LUpperArmAV', 'LForearmAV', 'ClubhandleAV',
%                'EquivalentMidpointCoupleLocal', 'AoA', and position/vector
%                components for 'Butt', 'RW', 'CH', 'Gripdx/dy/dz', 'Shaftdx/dy/dz'.
%
%   Output:
%       SummaryTable              - Table summarizing key values (max speeds, event times).
%       ClubQuiverAlphaReversal   - Struct with data for club/shaft quiver at Base Alpha Reversal time.
%       ClubQuiverMaxCHS          - Struct with data for club/shaft quiver at Base Max CHS time.
%       ClubQuiverZTCFAlphaReversal - Struct with data for club/shaft quiver at ZTCF Alpha Reversal time.
%       ClubQuiverDELTAAlphaReversal - Struct with data for club/shaft quiver at DELTA Alpha Reversal time.
%
%   Example:
%       [summary, qAR, qMaxCHS, qZTCF, qDELTA] = ...
%           generateSummaryTableAndQuiverData3D(BASEQ, ZTCFQ, DELTAQ);
%
%   See also: CALCULATEWORKPOWERANDANGULARIMPULSE3D

% Input validation
arguments
    BASEQ table {mustBeNonempty}
    ZTCFQ table {mustBeNonempty}
    DELTAQ table {mustBeNonempty}
end

% Validate all tables have Time column
assert(ismember('Time', BASEQ.Properties.VariableNames), ...
    'BASEQ table must contain a ''Time'' column');
assert(ismember('Time', ZTCFQ.Properties.VariableNames), ...
    'ZTCFQ table must contain a ''Time'' column');
assert(ismember('Time', DELTAQ.Properties.VariableNames), ...
    'DELTAQ table must contain a ''Time'' column');

% Validate Time columns are numeric and have no NaN
assert(isnumeric(BASEQ.Time) && all(~isnan(BASEQ.Time)), ...
    'BASEQ.Time must be numeric and contain no NaN values');
assert(isnumeric(ZTCFQ.Time) && all(~isnan(ZTCFQ.Time)), ...
    'ZTCFQ.Time must be numeric and contain no NaN values');
assert(isnumeric(DELTAQ.Time) && all(~isnan(DELTAQ.Time)), ...
    'DELTAQ.Time must be numeric and contain no NaN values');

% Validate tables have same length (same time grid)
assert(height(BASEQ) == height(ZTCFQ), ...
    'BASEQ and ZTCFQ tables must have the same number of rows');
assert(height(BASEQ) == height(DELTAQ), ...
    'BASEQ and DELTAQ tables must have the same number of rows');

% Validate tables have sufficient data
assert(height(BASEQ) >= 2, ...
    'Input tables must have at least 2 rows for meaningful analysis');

% Initialize outputs
SummaryTable = table();
ClubQuiverMaxCHS = struct();
ClubQuiverAlphaReversal = struct();
ClubQuiverZTCFAlphaReversal = struct();
ClubQuiverDELTAAlphaReversal = struct();

% --- Calculate Maximum Speeds and the Times They Occur (from BASEQ) ---

% Check if required columns exist in BASEQ for speed calculations
requiredSpeedCols_BASEQ = {'CHS (mph)', 'Hand Speed (mph)', 'BaseAV', 'ChestAV', 'LScapAV', 'LUpperArmAV', 'LForearmAV', 'ClubhandleAV'};
if all(ismember(requiredSpeedCols_BASEQ, BASEQ.Properties.VariableNames))

    % Generate CHS Array
    CHS = BASEQ{:, "CHS (mph)"};
    % Find Max CHS Value
    MaxCHS = max(CHS);
    SummaryTable.("MaxCHS") = MaxCHS;

    % Generate Hand Speed Array
    HS = BASEQ{:, "Hand Speed (mph)"};
    % Find Max Hand Speed Value
    MaxHandSpeed = max(HS);
    SummaryTable.("MaxHandSpeed") = MaxHandSpeed;

    % Generate Hip Angular Velocity (AV) Array
    HipAV = BASEQ{:, "BaseAV"};
    % Find Max Hip AV Value
    MaxHipAV = max(HipAV);
    SummaryTable.("MaxHipAV") = MaxHipAV;

    % Generate Torso Angular Velocity (AV) Array
    TorsoAV = BASEQ{:, "ChestAV"};
    % Find Max Torso AV Value
    MaxTorsoAV = max(TorsoAV);
    SummaryTable.("MaxTorsoAV") = MaxTorsoAV;

    % Generate LScap Angular Velocity (AV) Array
    LScapAV = BASEQ{:, "LScapAV"};
    % Find Max LScap AV Value
    MaxLScapAV = max(LScapAV);
    SummaryTable.("MaxLScapAV") = MaxLScapAV;

    % Generate LUpperArm Angular Velocity (AV) Array
    LUpperArmAV = BASEQ{:, "LUpperArmAV"};
    % Find Max LUpperArm AV Value
    MaxLUpperArmAV = max(LUpperArmAV);
    SummaryTable.("MaxLUpperArmAV") = MaxLUpperArmAV;

    % Generate LForearm Angular Velocity (AV) Array
    LForearmAV = BASEQ{:, "LForearmAV"};
    % Find Max LForearm AV Value
    MaxLForearmAV = max(LForearmAV);
    SummaryTable.("MaxLForearmAV") = MaxLForearmAV;

    % Generate Club Angular Velocity (AV) Array
    ClubAV = BASEQ{:, "ClubhandleAV"};
    % Find Max Club AV Value
    MaxClubAV = max(ClubAV);
    SummaryTable.("MaxClubAV") = MaxClubAV;

    % Find the row in the table where each maximum occurs
    CHSMaxRow = find(CHS == MaxCHS, 1);
    HSMaxRow = find(HS == MaxHandSpeed, 1);
    HipAVMaxRow = find(HipAV == MaxHipAV, 1);
    TorsoAVMaxRow = find(TorsoAV == MaxTorsoAV, 1);
    LScapAVMaxRow = find(LScapAV == MaxLScapAV, 1);
    LUpperArmAVMaxRow = find(LUpperArmAV == MaxLUpperArmAV, 1);
    LForearmAVMaxRow = find(LForearmAV == MaxLForearmAV, 1);
    ClubAVMaxRow = find(ClubAV == MaxClubAV, 1);

    % Find the time in the table where the maximum occurs
    CHSMaxTime = BASEQ.Time(CHSMaxRow, 1);
    SummaryTable.("CHSMaxTime") = CHSMaxTime;

    HandSpeedMaxTime = BASEQ.Time(HSMaxRow, 1);
    SummaryTable.("HandSpeedMaxTime") = HandSpeedMaxTime;

    HipAVMaxTime = BASEQ.Time(HipAVMaxRow, 1);
    SummaryTable.("HipAVMaxTime") = HipAVMaxTime;

    TorsoAVMaxTime = BASEQ.Time(TorsoAVMaxRow, 1);
    SummaryTable.("TorsoAVMaxTime") = TorsoAVMaxTime;

    LScapAVMaxTime = BASEQ.Time(LScapAVMaxRow, 1);
    SummaryTable.("LScapAVMaxTime") = LScapAVMaxTime;

    LUpperArmAVMaxTime = BASEQ.Time(LUpperArmAVMaxRow, 1);
    SummaryTable.("LUpperArmAVMaxTime") = LUpperArmAVMaxTime;

    LForearmAVMaxTime = BASEQ.Time(LForearmAVMaxRow, 1);
    SummaryTable.("LForearmAVMaxTime") = LForearmAVMaxTime;

    ClubAVMaxTime = BASEQ.Time(ClubAVMaxRow, 1);
    SummaryTable.("MaxClubAVTime") = ClubAVMaxTime; % Renamed from ClubAVMaxTime for clarity

    % Find AoA at time of maximum CHS
    if ismember('AoA', BASEQ.Properties.VariableNames)
        AoAatMaxCHS = BASEQ.AoA(CHSMaxRow, 1);
        SummaryTable.("AoAatMaxCHS") = AoAatMaxCHS;
    else
        warning('Required data column "AoA" not found in BASEQ for AoAatMaxCHS.');
        SummaryTable.("AoAatMaxCHS") = NaN;
    end

else
    warning('Required columns for speed calculations not found in BASEQ.');
    % Initialize speed/time fields in SummaryTable to NaN
    speedFields = {'MaxCHS', 'MaxHandSpeed', 'HipAV', 'TorsoAV', 'LScapAV', 'LUpperArmAV', 'LForearmAV', 'ClubAV'};
    timeFields = {'CHSMaxTime', 'HandSpeedMaxTime', 'HipAVMaxTime', 'TorsoAVMaxTime', 'LScapAVMaxTime', 'LUpperArmAVMaxTime', 'LForearmAVMaxTime', 'MaxClubAVTime'};
     for k = 1:length(speedFields)
         SummaryTable.(speedFields{k}) = NaN;
     end
     for k = 1:length(timeFields)
         SummaryTable.(timeFields{k}) = NaN;
     end
     SummaryTable.("AoAatMaxCHS") = NaN;
end


% --- Calculate Times of Specific Events ---

% Calculate the time that the equivalent midpoint couple goes negative in late downswing (BaseData)
% Interpolate where the 3rd component of EquivalentMidpointCoupleLocal crosses zero.
% Starting interpolation from index 50 as in original script to avoid startup effects.
% Ensure the data column exists and has enough points.
if ismember('EquivalentMidpointCoupleLocal', BASEQ.Properties.VariableNames) && height(BASEQ) >= 50
    timeData = BASEQ.Time(50:end, 1);
    coupleData = BASEQ.EquivalentMidpointCoupleLocal(50:end, 3); % Assuming 3rd component is relevant
    % Check if interpolation range is valid (at least 2 points) and data is not constant
    if length(timeData) >= 2 && range(coupleData) > 0
         TimeofAlphaReversal = interp1(coupleData, timeData, 0.0, 'linear', 'extrap'); % Use 'extrap' if zero is outside range
         SummaryTable.("TimeofAlphaReversal") = TimeofAlphaReversal;
    else
         warning('Not enough varying data points to calculate TimeofAlphaReversal for BASEQ.');
         SummaryTable.("TimeofAlphaReversal") = NaN;
    end
else
    warning('Required data column "EquivalentMidpointCoupleLocal" not found or not enough data in BASEQ for TimeofAlphaReversal.');
    SummaryTable.("TimeofAlphaReversal") = NaN;
end

% Calculate the time that the ZTCF equivalent midpoint couple goes negative.
if ismember('EquivalentMidpointCoupleLocal', ZTCFQ.Properties.VariableNames) && height(ZTCFQ) >= 50
     timeData_ZTCF = ZTCFQ.Time(50:end, 1);
     coupleData_ZTCF = ZTCFQ.EquivalentMidpointCoupleLocal(50:end, 3);
     if length(timeData_ZTCF) >= 2 && range(coupleData_ZTCF) > 0
        TimeofZTCFAlphaReversal = interp1(coupleData_ZTCF, timeData_ZTCF, 0.0, 'linear', 'extrap');
        SummaryTable.("TimeofZTCFAlphaReversal") = TimeofZTCFAlphaReversal;
     else
        warning('Not enough varying data points to calculate TimeofZTCFAlphaReversal for ZTCFQ.');
        SummaryTable.("TimeofZTCFAlphaReversal") = NaN;
     end
else
    warning('Required data column "EquivalentMidpointCoupleLocal" not found or not enough data in ZTCFQ for TimeofZTCFAlphaReversal.');
    SummaryTable.("TimeofZTCFAlphaReversal") = NaN;
end

% Calculate the time that the DELTA equivalent midpoint couple goes negative.
if ismember('EquivalentMidpointCoupleLocal', DELTAQ.Properties.VariableNames) && height(DELTAQ) >= 50
    timeData_DELTA = DELTAQ.Time(50:end, 1);
    coupleData_DELTA = DELTAQ.EquivalentMidpointCoupleLocal(50:end, 3);
    if length(timeData_DELTA) >= 2 && range(coupleData_DELTA) > 0
        TimeofDELTAAlphaReversal = interp1(coupleData_DELTA, timeData_DELTA, 0.0, 'linear', 'extrap');
        SummaryTable.("TimeofDELTAAlphaReversal") = TimeofDELTAAlphaReversal;
    else
        warning('Not enough varying data points to calculate TimeofDELTAAlphaReversal for DELTAQ.');
        SummaryTable.("TimeofDELTAAlphaReversal") = NaN;
    end
else
    warning('Required data column "EquivalentMidpointCoupleLocal" not found or not enough data in DELTAQ for TimeofDELTAAlphaReversal.');
    SummaryTable.("TimeofDELTAAlphaReversal") = NaN;
end


% Generate a table of the times when the function of interest (f) crosses zero (for AoA).
% Assumes 'AoA' column exists in BASEQ.
if ismember('AoA', BASEQ.Properties.VariableNames)
    f = BASEQ.AoA;
    t = BASEQ.Time;

    % Find indices where the sign changes
    idx = find(f(2:end) .* f(1:end-1) < 0);

    % Interpolate to find the exact time of zero crossing for each sign change
    t_zero = zeros(size(idx));
    for i = 1:numel(idx)
        j = idx(i); % Index in the original vector where sign change starts
        % Interpolate between point j and j+1
        t_zero(i) = interp1(f(j:j+1), t(j:j+1), 0.0, 'linear');
    end

    % Time of Zero AoA that Occurs Last (assuming downswing is the relevant phase)
    if ~isempty(t_zero)
        TimeofZeroAoA = max(t_zero); % Find the latest zero crossing
        SummaryTable.("TimeofZeroAoA") = TimeofZeroAoA;

        % CHS at time of zero AoA (interpolate CHS at this specific time)
        if ismember('CHS (mph)', BASEQ.Properties.VariableNames) && ~isnan(TimeofZeroAoA)
             CHSZeroAoA = interp1(BASEQ.Time, BASEQ.("CHS (mph)"), TimeofZeroAoA, 'linear', 'extrap');
             SummaryTable.("CHSZeroAoA") = CHSZeroAoA;
        else
             warning('Required data column "CHS (mph)" not found in BASEQ or TimeofZeroAoA invalid for CHSZeroAoA.');
             SummaryTable.("CHSZeroAoA") = NaN;
        end
    else
        warning('No zero crossing found for AoA in BASEQ.');
        SummaryTable.("TimeofZeroAoA") = NaN;
        SummaryTable.("CHSZeroAoA") = NaN;
    end
else
    warning('Required data column "AoA" not found in BASEQ.');
    SummaryTable.("TimeofZeroAoA") = NaN;
    SummaryTable.("CHSZeroAoA") = NaN;
end


% --- Generate Data Needed for Quivers at Specific Times ---
% These structs will hold position and vector data at key moments for plotting.
% Interpolate data from BASEQ, ZTCFQ, DELTAQ at the calculated times.

% Required columns for quiver data generation
requiredQuiverCols = {'Buttx', 'Butty', 'Buttz', 'Gripdx', 'Gripdy', 'Gripdz', 'RWx', 'RWy', 'RWz', 'Shaftdx', 'Shaftdy', 'Shaftdz'};


% Find Data Needed for Grip and Shaft Quivers at Time of Max CHS (from BASEQ)
if isfield(SummaryTable, 'CHSMaxTime') && ~isnan(SummaryTable.CHSMaxTime) && all(ismember(requiredQuiverCols, BASEQ.Properties.VariableNames))
    t_maxCHS = SummaryTable.CHSMaxTime;
    ClubQuiverMaxCHS.("ButtxMaxCHS") = interp1(BASEQ.Time, BASEQ.("Buttx"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("ButtyMaxCHS") = interp1(BASEQ.Time, BASEQ.("Butty"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("ButtzMaxCHS") = interp1(BASEQ.Time, BASEQ.("Buttz"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("GripdxMaxCHS") = interp1(BASEQ.Time, BASEQ.("Gripdx"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("GripdyMaxCHS") = interp1(BASEQ.Time, BASEQ.("Gripdy"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("GripdzMaxCHS") = interp1(BASEQ.Time, BASEQ.("Gripdz"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("RWxMaxCHS") = interp1(BASEQ.Time, BASEQ.("RWx"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("RWyMaxCHS") = interp1(BASEQ.Time, BASEQ.("RWy"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("RWzMaxCHS") = interp1(BASEQ.Time, BASEQ.("RWz"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("ShaftdxMaxCHS") = interp1(BASEQ.Time, BASEQ.("Shaftdx"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("ShaftdyMaxCHS") = interp1(BASEQ.Time, BASEQ.("Shaftdy"), t_maxCHS, 'linear', 'extrap');
    ClubQuiverMaxCHS.("ShaftdzMaxCHS") = interp1(BASEQ.Time, BASEQ.("Shaftdz"), t_maxCHS, 'linear', 'extrap');
else
    warning('Cannot generate ClubQuiverMaxCHS data: CHSMaxTime invalid or required columns missing in BASEQ.');
    % Initialize fields to NaN
    for k = 1:length(requiredQuiverCols)
        ClubQuiverMaxCHS.(requiredQuiverCols{k} + "MaxCHS") = NaN;
    end
end


% Find Data Needed for Grip and Shaft Quivers at Time of Alpha Reversal (from BASEQ)
if isfield(SummaryTable, 'TimeofAlphaReversal') && ~isnan(SummaryTable.TimeofAlphaReversal) && all(ismember(requiredQuiverCols, BASEQ.Properties.VariableNames))
    t_alphaRev = SummaryTable.TimeofAlphaReversal;
    ClubQuiverAlphaReversal.("ButtxAlphaReversal") = interp1(BASEQ.Time, BASEQ.("Buttx"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("ButtyAlphaReversal") = interp1(BASEQ.Time, BASEQ.("Butty"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("ButtzAlphaReversal") = interp1(BASEQ.Time, BASEQ.("Buttz"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("GripdxAlphaReversal") = interp1(BASEQ.Time, BASEQ.("Gripdx"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("GripdyAlphaReversal") = interp1(BASEQ.Time, BASEQ.("Gripdy"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("GripdzAlphaReversal") = interp1(BASEQ.Time, BASEQ.("Gripdz"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("RWxAlphaReversal") = interp1(BASEQ.Time, BASEQ.("RWx"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("RWyAlphaReversal") = interp1(BASEQ.Time, BASEQ.("RWy"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("RWzAlphaReversal") = interp1(BASEQ.Time, BASEQ.("RWz"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("ShaftdxAlphaReversal") = interp1(BASEQ.Time, BASEQ.("Shaftdx"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("ShaftdyAlphaReversal") = interp1(BASEQ.Time, BASEQ.("Shaftdy"), t_alphaRev, 'linear', 'extrap');
    ClubQuiverAlphaReversal.("ShaftdzAlphaReversal") = interp1(BASEQ.Time, BASEQ.("Shaftdz"), t_alphaRev, 'linear', 'extrap');
else
    warning('Cannot generate ClubQuiverAlphaReversal data: TimeofAlphaReversal invalid or required columns missing in BASEQ.');
     % Initialize fields to NaN
     for k = 1:length(requiredQuiverCols)
        ClubQuiverAlphaReversal.(requiredQuiverCols{k} + "AlphaReversal") = NaN;
     end
end


% Find Data Needed for Grip and Shaft Quivers at Time of ZTCF Alpha Reversal (from ZTCFQ)
if isfield(SummaryTable, 'TimeofZTCFAlphaReversal') && ~isnan(SummaryTable.TimeofZTCFAlphaReversal) && all(ismember(requiredQuiverCols, ZTCFQ.Properties.VariableNames))
    t_ztcfAlphaRev = SummaryTable.TimeofZTCFAlphaReversal;
    ClubQuiverZTCFAlphaReversal.("ButtxZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("Buttx"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("ButtyZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("Butty"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("ButtzZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("Buttz"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("GripdxZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("Gripdx"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("GripdyZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("Gripdy"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("GripdzZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("Gripdz"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("RWxZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("RWx"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("RWyZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("RWy"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("RWzZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("RWz"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("ShaftdxZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("Shaftdx"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("ShaftdyZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("Shaftdy"), t_ztcfAlphaRev, 'linear', 'extrap');
    ClubQuiverZTCFAlphaReversal.("ShaftdzZTCFAlphaReversal") = interp1(ZTCFQ.Time, ZTCFQ.("Shaftdz"), t_ztcfAlphaRev, 'linear', 'extrap');
else
    warning('Cannot generate ClubQuiverZTCFAlphaReversal data: TimeofZTCFAlphaReversal invalid or required columns missing in ZTCFQ.');
     % Initialize fields to NaN
     for k = 1:length(requiredQuiverCols)
        ClubQuiverZTCFAlphaReversal.(requiredQuiverCols{k} + "ZTCFAlphaReversal") = NaN;
     end
end


% Find Data Needed for Grip and Shaft Quivers at Time of DELTA Alpha Reversal (from DELTAQ)
if isfield(SummaryTable, 'TimeofDELTAAlphaReversal') && ~isnan(SummaryTable.TimeofDELTAAlphaReversal) && all(ismember(requiredQuiverCols, DELTAQ.Properties.VariableNames))
    t_deltaAlphaRev = SummaryTable.TimeofDELTAAlphaReversal;
    ClubQuiverDELTAAlphaReversal.("ButtxDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("Buttx"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("ButtyDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("Butty"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("ButtzDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("Buttz"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("GripdxDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("Gripdx"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("GripdyDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("Gripdy"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("GripdzDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("Gripdz"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("RWxDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("RWx"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("RWyDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("RWy"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("RWzDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("RWz"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("ShaftdxDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("Shaftdx"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("ShaftdyDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("Shaftdy"), t_deltaAlphaRev, 'linear', 'extrap');
    ClubQuiverDELTAAlphaReversal.("ShaftdzDELTAAlphaReversal") = interp1(DELTAQ.Time, DELTAQ.("Shaftdz"), t_deltaAlphaRev, 'linear', 'extrap');
else
    warning('Cannot generate ClubQuiverDELTAAlphaReversal data: TimeofDELTAAlphaReversal invalid or required columns missing in DELTAQ.');
     % Initialize fields to NaN
     for k = 1:length(requiredQuiverCols)
        ClubQuiverDELTAAlphaReversal.(requiredQuiverCols{k} + "DELTAAlphaReversal") = NaN;
     end
end


% No need for explicit 'clear' statements within a function.
% All intermediate variables are local and cleared automatically.

end
