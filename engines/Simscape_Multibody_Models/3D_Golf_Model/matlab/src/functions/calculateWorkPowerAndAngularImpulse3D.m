function [ZTCFQ_updated, DELTAQ_updated] = calculateWorkPowerAndAngularImpulse3D(ZTCFQ, DELTAQ, options)
% CALCULATEWORKPOWERANDANGULARIMPULSE3D Enhanced work, power, and angular impulse calculations.
%   [ZTCFQ_updated, DELTAQ_updated] = CALCULATEWORKPOWERANDANGULARIMPULSE3D(ZTCFQ, DELTAQ, options)
%   Calculates linear power, optional linear work, angular power, optional angular work,
%   and total angular impulse (including applied torques and force moments) for various
%   body segments based on the data in the ZTCFQ and DELTAQ tables.
%
%   Input:
%       ZTCFQ  - Table containing ZTCF data on a uniform time grid
%       DELTAQ - Table containing Delta data on the same uniform time grid
%       options - Structure with calculation options:
%           .calculate_work (logical) - Whether to calculate work (default: false)
%           .include_applied_torques (logical) - Include applied torques in angular impulse (default: true)
%           .include_force_moments (logical) - Include force moments in angular impulse (default: true)
%
%   Output:
%       ZTCFQ_updated  - Updated ZTCFQ table with added power, work (optional), and angular impulse columns
%       DELTAQ_updated - Updated DELTAQ table with added power, work (optional), and angular impulse columns
%
%   Example:
%       options = struct('calculate_work', true);
%       [ZTCFQ_new, DELTAQ_new] = calculateWorkPowerAndAngularImpulse3D(ZTCFQ, DELTAQ, options);
%
%   See also: GENERATESUMMARYTABLEANDQUIVERDATA3D

% Input validation
arguments
    ZTCFQ table {mustBeNonempty}
    DELTAQ table {mustBeNonempty}
    options struct = struct()
end

% Validate tables have required Time column
assert(ismember('Time', ZTCFQ.Properties.VariableNames), ...
    'ZTCFQ table must contain a ''Time'' column');
assert(ismember('Time', DELTAQ.Properties.VariableNames), ...
    'DELTAQ table must contain a ''Time'' column');

% Validate Time columns contain numeric data and no NaN values
assert(isnumeric(ZTCFQ.Time) && all(~isnan(ZTCFQ.Time)), ...
    'ZTCFQ.Time must be numeric and contain no NaN values');
assert(isnumeric(DELTAQ.Time) && all(~isnan(DELTAQ.Time)), ...
    'DELTAQ.Time must be numeric and contain no NaN values');

% Validate tables have same length (same time grid)
assert(height(ZTCFQ) == height(DELTAQ), ...
    'ZTCFQ and DELTAQ tables must have the same number of rows (same time grid)');

% Validate tables have at least 2 rows (needed for time step calculation)
assert(height(ZTCFQ) >= 2, ...
    'Input tables must have at least 2 rows to calculate work/power/impulse');

% Set default options
if ~isfield(options, 'calculate_work')
    options.calculate_work = false;
end
if ~isfield(options, 'include_applied_torques')
    options.include_applied_torques = true;
end
if ~isfield(options, 'include_force_moments')
    options.include_force_moments = true;
end

% Assign input tables to internal variables for modification
ZTCFQ_updated = ZTCFQ;
DELTAQ_updated = DELTAQ;

% --- Get Time Step (Assuming Uniform Grid) ---
if height(ZTCFQ_updated) > 1
    TsQ = ZTCFQ_updated.Time(2) - ZTCFQ_updated.Time(1);
else
    warning('Input table has less than 2 rows. Cannot calculate work/impulse.');
    return;
end

% --- Process ZTCFQ Table ---
fprintf('Calculating power, work (optional), and angular impulse for ZTCFQ...\n');

try
    % Extract time vector
    time_data = ZTCFQ_updated.Time;

    % Extract forces (Global frame, expected as Nx3)
    % Based on the CSV structure from trial_001_20250802_204903.csv
    F_ZTCF = ZTCFQ_updated{:, ["CalculatedSignalsLogs_TotalHandForceGlobal_1", ...
                               "CalculatedSignalsLogs_TotalHandForceGlobal_2", ...
                               "CalculatedSignalsLogs_TotalHandForceGlobal_3"]};
    LHF_ZTCF = ZTCFQ_updated{:, ["CalculatedSignalsLogs_LHonClubForceGlobal_1", ...
                                 "CalculatedSignalsLogs_LHonClubForceGlobal_2", ...
                                 "CalculatedSignalsLogs_LHonClubForceGlobal_3"]};
    RHF_ZTCF = ZTCFQ_updated{:, ["CalculatedSignalsLogs_RHonClubForceGlobal_1", ...
                                 "CalculatedSignalsLogs_RHonClubForceGlobal_2", ...
                                 "CalculatedSignalsLogs_RHonClubForceGlobal_3"]};

    % Extract velocities (Global frame, expected as Nx3)
    V_ZTCF = ZTCFQ_updated{:, ["MidpointCalcsLogs_MPGlobalVelocity_1", ...
                               "MidpointCalcsLogs_MPGlobalVelocity_2", ...
                               "MidpointCalcsLogs_MPGlobalVelocity_3"]};
    LHV_ZTCF = ZTCFQ_updated{:, ["LHCalcsLogs_LHGlobalVelocity_1", ...
                                 "LHCalcsLogs_LHGlobalVelocity_2", ...
                                 "LHCalcsLogs_LHGlobalVelocity_3"]};
    RHV_ZTCF = ZTCFQ_updated{:, ["RHCalcsLogs_RHGlobalVelocity_1", ...
                                 "RHCalcsLogs_RHGlobalVelocity_2", ...
                                 "RHCalcsLogs_RHGlobalVelocity_3"]};

    % Extract angular velocities (Global frame, expected as Nx3)
    LHAV_ZTCF = ZTCFQ_updated{:, ["LWLogs_LHGlobalAngularVelocity_1", ...
                                  "LWLogs_LHGlobalAngularVelocity_2", ...
                                  "LWLogs_LHGlobalAngularVelocity_3"]};
    RHAV_ZTCF = ZTCFQ_updated{:, ["RWLogs_RHGlobalAngularVelocity_1", ...
                                  "RWLogs_RHGlobalAngularVelocity_2", ...
                                  "RWLogs_RHGlobalAngularVelocity_3"]};

    % Extract torques (Global frame, expected as Nx3)
    LH_Torque_ZTCF = ZTCFQ_updated{:, ["CalculatedSignalsLogs_TotalHandTorqueGlobal_1", ...
                                       "CalculatedSignalsLogs_TotalHandTorqueGlobal_2", ...
                                       "CalculatedSignalsLogs_TotalHandTorqueGlobal_3"]};
    RH_Torque_ZTCF = ZTCFQ_updated{:, ["CalculatedSignalsLogs_TotalHandTorqueGlobal_1", ...
                                       "CalculatedSignalsLogs_TotalHandTorqueGlobal_2", ...
                                       "CalculatedSignalsLogs_TotalHandTorqueGlobal_3"]};

    % Extract applied torques (for angular impulse calculation)
    % These are the actuator torques from the joints
    LS_Applied_Torque_ZTCF = ZTCFQ_updated{:, ["LSLogs_ActuatorTorqueX", ...
                                               "LSLogs_ActuatorTorqueY", ...
                                               "LSLogs_ActuatorTorqueZ"]};
    RS_Applied_Torque_ZTCF = ZTCFQ_updated{:, ["RSLogs_ActuatorTorqueX", ...
                                               "RSLogs_ActuatorTorqueY", ...
                                               "RSLogs_ActuatorTorqueZ"]};
    LE_Applied_Torque_ZTCF = ZTCFQ_updated{:, ["LELogs_ActuatorTorque", ...
                                               "LELogs_ActuatorTorque", ...
                                               "LELogs_ActuatorTorque"]}; % Single value repeated
    RE_Applied_Torque_ZTCF = ZTCFQ_updated{:, ["RELogs_ActuatorTorque", ...
                                               "RELogs_ActuatorTorque", ...
                                               "RELogs_ActuatorTorque"]}; % Single value repeated

    % Extract force moments (for angular impulse calculation)
    % These are the moments of forces at the joints
    LS_Force_Moment_ZTCF = ZTCFQ_updated{:, ["LSLogs_TorqueLocal_1", ...
                                             "LSLogs_TorqueLocal_2", ...
                                             "LSLogs_TorqueLocal_3"]};
    RS_Force_Moment_ZTCF = ZTCFQ_updated{:, ["RSLogs_TorqueLocal_1", ...
                                             "RSLogs_TorqueLocal_2", ...
                                             "RSLogs_TorqueLocal_3"]};
    LE_Force_Moment_ZTCF = ZTCFQ_updated{:, ["LELogs_LArmonLForearmTGlobal_1", ...
                                             "LELogs_LArmonLForearmTGlobal_2", ...
                                             "LELogs_LArmonLForearmTGlobal_3"]};
    RE_Force_Moment_ZTCF = ZTCFQ_updated{:, ["RELogs_RArmonLForearmTGlobal_1", ...
                                             "RELogs_RArmonLForearmTGlobal_2", ...
                                             "RELogs_RArmonLForearmTGlobal_3"]};

    % Calculate linear power (P = F · v)
    LH_Linear_Power_ZTCF = dot(LHF_ZTCF, LHV_ZTCF, 2);
    RH_Linear_Power_ZTCF = dot(RHF_ZTCF, RHV_ZTCF, 2);
    Total_Linear_Power_ZTCF = LH_Linear_Power_ZTCF + RH_Linear_Power_ZTCF;

    % Calculate angular power (P = τ · ω)
    LH_Angular_Power_ZTCF = dot(LH_Torque_ZTCF, LHAV_ZTCF, 2);
    RH_Angular_Power_ZTCF = dot(RH_Torque_ZTCF, RHAV_ZTCF, 2);
    Total_Angular_Power_ZTCF = LH_Angular_Power_ZTCF + RH_Angular_Power_ZTCF;

    % Calculate work if requested (W = ∫ P dt)
    if options.calculate_work
        LH_Linear_Work_ZTCF = cumtrapz(time_data, LH_Linear_Power_ZTCF);
        RH_Linear_Work_ZTCF = cumtrapz(time_data, RH_Linear_Power_ZTCF);
        Total_Linear_Work_ZTCF = LH_Linear_Work_ZTCF + RH_Linear_Work_ZTCF;

        LH_Angular_Work_ZTCF = cumtrapz(time_data, LH_Angular_Power_ZTCF);
        RH_Angular_Work_ZTCF = cumtrapz(time_data, RH_Angular_Power_ZTCF);
        Total_Angular_Work_ZTCF = LH_Angular_Work_ZTCF + RH_Angular_Work_ZTCF;
    end

    % Calculate total angular impulse (∫ τ_total dt)
    % This includes applied torques and force moments
    Total_Angular_Impulse_ZTCF = zeros(size(time_data, 1), 3);

    if options.include_applied_torques
        % Add applied torques to angular impulse
        Total_Angular_Impulse_ZTCF = Total_Angular_Impulse_ZTCF + ...
            cumtrapz(time_data, LS_Applied_Torque_ZTCF) + ...
            cumtrapz(time_data, RS_Applied_Torque_ZTCF) + ...
            cumtrapz(time_data, LE_Applied_Torque_ZTCF) + ...
            cumtrapz(time_data, RE_Applied_Torque_ZTCF);
    end

    if options.include_force_moments
        % Add force moments to angular impulse
        Total_Angular_Impulse_ZTCF = Total_Angular_Impulse_ZTCF + ...
            cumtrapz(time_data, LS_Force_Moment_ZTCF) + ...
            cumtrapz(time_data, RS_Force_Moment_ZTCF) + ...
            cumtrapz(time_data, LE_Force_Moment_ZTCF) + ...
            cumtrapz(time_data, RE_Force_Moment_ZTCF);
    end

    % Add calculated columns to ZTCFQ_updated
    ZTCFQ_updated.LH_Linear_Power = LH_Linear_Power_ZTCF;
    ZTCFQ_updated.RH_Linear_Power = RH_Linear_Power_ZTCF;
    ZTCFQ_updated.Total_Linear_Power = Total_Linear_Power_ZTCF;

    ZTCFQ_updated.LH_Angular_Power = LH_Angular_Power_ZTCF;
    ZTCFQ_updated.RH_Angular_Power = RH_Angular_Power_ZTCF;
    ZTCFQ_updated.Total_Angular_Power = Total_Angular_Power_ZTCF;

    if options.calculate_work
        ZTCFQ_updated.LH_Linear_Work = LH_Linear_Work_ZTCF;
        ZTCFQ_updated.RH_Linear_Work = RH_Linear_Work_ZTCF;
        ZTCFQ_updated.Total_Linear_Work = Total_Linear_Work_ZTCF;

        ZTCFQ_updated.LH_Angular_Work = LH_Angular_Work_ZTCF;
        ZTCFQ_updated.RH_Angular_Work = RH_Angular_Work_ZTCF;
        ZTCFQ_updated.Total_Angular_Work = Total_Angular_Work_ZTCF;
    end

    ZTCFQ_updated.Total_Angular_Impulse_X = Total_Angular_Impulse_ZTCF(:, 1);
    ZTCFQ_updated.Total_Angular_Impulse_Y = Total_Angular_Impulse_ZTCF(:, 2);
    ZTCFQ_updated.Total_Angular_Impulse_Z = Total_Angular_Impulse_ZTCF(:, 3);

catch ME
    warning('Error processing ZTCFQ data: %s', ME.message);
    % Return the original tables as is if columns are missing
    ZTCFQ_updated = ZTCFQ;
    DELTAQ_updated = DELTAQ;
    return;
end

% --- Process DELTAQ Table ---
fprintf('Calculating power, work (optional), and angular impulse for DELTAQ...\n');

try
    % Extract time vector
    time_data = DELTAQ_updated.Time;

    % Extract forces for DELTAQ (similar structure but for delta calculations)
    % Note: DELTAQ typically contains the difference between two conditions
    % The column names might be similar but represent delta values

    % For now, we'll use the same structure as ZTCFQ but with delta prefixes
    % In practice, you'd need to adjust these based on your actual DELTAQ structure

    % Extract forces (Global frame, expected as Nx3)
    F_DELTAQ = DELTAQ_updated{:, ["CalculatedSignalsLogs_TotalHandForceGlobal_1", ...
                                 "CalculatedSignalsLogs_TotalHandForceGlobal_2", ...
                                 "CalculatedSignalsLogs_TotalHandForceGlobal_3"]};
    LHF_DELTAQ = DELTAQ_updated{:, ["CalculatedSignalsLogs_LHonClubForceGlobal_1", ...
                                   "CalculatedSignalsLogs_LHonClubForceGlobal_2", ...
                                   "CalculatedSignalsLogs_LHonClubForceGlobal_3"]};
    RHF_DELTAQ = DELTAQ_updated{:, ["CalculatedSignalsLogs_RHonClubForceGlobal_1", ...
                                   "CalculatedSignalsLogs_RHonClubForceGlobal_2", ...
                                   "CalculatedSignalsLogs_RHonClubForceGlobal_3"]};

    % Extract velocities (Global frame, expected as Nx3)
    V_DELTAQ = DELTAQ_updated{:, ["MidpointCalcsLogs_MPGlobalVelocity_1", ...
                                 "MidpointCalcsLogs_MPGlobalVelocity_2", ...
                                 "MidpointCalcsLogs_MPGlobalVelocity_3"]};
    LHV_DELTAQ = DELTAQ_updated{:, ["LHCalcsLogs_LHGlobalVelocity_1", ...
                                   "LHCalcsLogs_LHGlobalVelocity_2", ...
                                   "LHCalcsLogs_LHGlobalVelocity_3"]};
    RHV_DELTAQ = DELTAQ_updated{:, ["RHCalcsLogs_RHGlobalVelocity_1", ...
                                   "RHCalcsLogs_RHGlobalVelocity_2", ...
                                   "RHCalcsLogs_RHGlobalVelocity_3"]};

    % Extract angular velocities (Global frame, expected as Nx3)
    LHAV_DELTAQ = DELTAQ_updated{:, ["LWLogs_LHGlobalAngularVelocity_1", ...
                                    "LWLogs_LHGlobalAngularVelocity_2", ...
                                    "LWLogs_LHGlobalAngularVelocity_3"]};
    RHAV_DELTAQ = DELTAQ_updated{:, ["RWLogs_RHGlobalAngularVelocity_1", ...
                                    "RWLogs_RHGlobalAngularVelocity_2", ...
                                    "RWLogs_RHGlobalAngularVelocity_3"]};

    % Extract torques (Global frame, expected as Nx3)
    LH_Torque_DELTAQ = DELTAQ_updated{:, ["CalculatedSignalsLogs_TotalHandTorqueGlobal_1", ...
                                         "CalculatedSignalsLogs_TotalHandTorqueGlobal_2", ...
                                         "CalculatedSignalsLogs_TotalHandTorqueGlobal_3"]};
    RH_Torque_DELTAQ = DELTAQ_updated{:, ["CalculatedSignalsLogs_TotalHandTorqueGlobal_1", ...
                                         "CalculatedSignalsLogs_TotalHandTorqueGlobal_2", ...
                                         "CalculatedSignalsLogs_TotalHandTorqueGlobal_3"]};

    % Extract applied torques (for angular impulse calculation)
    LS_Applied_Torque_DELTAQ = DELTAQ_updated{:, ["LSLogs_ActuatorTorqueX", ...
                                                  "LSLogs_ActuatorTorqueY", ...
                                                  "LSLogs_ActuatorTorqueZ"]};
    RS_Applied_Torque_DELTAQ = DELTAQ_updated{:, ["RSLogs_ActuatorTorqueX", ...
                                                  "RSLogs_ActuatorTorqueY", ...
                                                  "RSLogs_ActuatorTorqueZ"]};
    LE_Applied_Torque_DELTAQ = DELTAQ_updated{:, ["LELogs_ActuatorTorque", ...
                                                  "LELogs_ActuatorTorque", ...
                                                  "LELogs_ActuatorTorque"]};
    RE_Applied_Torque_DELTAQ = DELTAQ_updated{:, ["RELogs_ActuatorTorque", ...
                                                  "RELogs_ActuatorTorque", ...
                                                  "RELogs_ActuatorTorque"]};

    % Extract force moments (for angular impulse calculation)
    LS_Force_Moment_DELTAQ = DELTAQ_updated{:, ["LSLogs_TorqueLocal_1", ...
                                                "LSLogs_TorqueLocal_2", ...
                                                "LSLogs_TorqueLocal_3"]};
    RS_Force_Moment_DELTAQ = DELTAQ_updated{:, ["RSLogs_TorqueLocal_1", ...
                                                "RSLogs_TorqueLocal_2", ...
                                                "RSLogs_TorqueLocal_3"]};
    LE_Force_Moment_DELTAQ = DELTAQ_updated{:, ["LELogs_LArmonLForearmTGlobal_1", ...
                                                "LELogs_LArmonLForearmTGlobal_2", ...
                                                "LELogs_LArmonLForearmTGlobal_3"]};
    RE_Force_Moment_DELTAQ = DELTAQ_updated{:, ["RELogs_RArmonLForearmTGlobal_1", ...
                                                "RELogs_RArmonLForearmTGlobal_2", ...
                                                "RELogs_RArmonLForearmTGlobal_3"]};

    % Calculate linear power (P = F · v)
    LH_Linear_Power_DELTAQ = dot(LHF_DELTAQ, LHV_DELTAQ, 2);
    RH_Linear_Power_DELTAQ = dot(RHF_DELTAQ, RHV_DELTAQ, 2);
    Total_Linear_Power_DELTAQ = LH_Linear_Power_DELTAQ + RH_Linear_Power_DELTAQ;

    % Calculate angular power (P = τ · ω)
    LH_Angular_Power_DELTAQ = dot(LH_Torque_DELTAQ, LHAV_DELTAQ, 2);
    RH_Angular_Power_DELTAQ = dot(RH_Torque_DELTAQ, RHAV_DELTAQ, 2);
    Total_Angular_Power_DELTAQ = LH_Angular_Power_DELTAQ + RH_Angular_Power_DELTAQ;

    % Calculate work if requested (W = ∫ P dt)
    if options.calculate_work
        LH_Linear_Work_DELTAQ = cumtrapz(time_data, LH_Linear_Power_DELTAQ);
        RH_Linear_Work_DELTAQ = cumtrapz(time_data, RH_Linear_Power_DELTAQ);
        Total_Linear_Work_DELTAQ = LH_Linear_Work_DELTAQ + RH_Linear_Work_DELTAQ;

        LH_Angular_Work_DELTAQ = cumtrapz(time_data, LH_Angular_Power_DELTAQ);
        RH_Angular_Work_DELTAQ = cumtrapz(time_data, RH_Angular_Power_DELTAQ);
        Total_Angular_Work_DELTAQ = LH_Angular_Work_DELTAQ + RH_Angular_Work_DELTAQ;
    end

    % Calculate total angular impulse (∫ τ_total dt)
    Total_Angular_Impulse_DELTAQ = zeros(size(time_data, 1), 3);

    if options.include_applied_torques
        % Add applied torques to angular impulse
        Total_Angular_Impulse_DELTAQ = Total_Angular_Impulse_DELTAQ + ...
            cumtrapz(time_data, LS_Applied_Torque_DELTAQ) + ...
            cumtrapz(time_data, RS_Applied_Torque_DELTAQ) + ...
            cumtrapz(time_data, LE_Applied_Torque_DELTAQ) + ...
            cumtrapz(time_data, RE_Applied_Torque_DELTAQ);
    end

    if options.include_force_moments
        % Add force moments to angular impulse
        Total_Angular_Impulse_DELTAQ = Total_Angular_Impulse_DELTAQ + ...
            cumtrapz(time_data, LS_Force_Moment_DELTAQ) + ...
            cumtrapz(time_data, RS_Force_Moment_DELTAQ) + ...
            cumtrapz(time_data, LE_Force_Moment_DELTAQ) + ...
            cumtrapz(time_data, RE_Force_Moment_DELTAQ);
    end

    % Add calculated columns to DELTAQ_updated
    DELTAQ_updated.LH_Linear_Power = LH_Linear_Power_DELTAQ;
    DELTAQ_updated.RH_Linear_Power = RH_Linear_Power_DELTAQ;
    DELTAQ_updated.Total_Linear_Power = Total_Linear_Power_DELTAQ;

    DELTAQ_updated.LH_Angular_Power = LH_Angular_Power_DELTAQ;
    DELTAQ_updated.RH_Angular_Power = RH_Angular_Power_DELTAQ;
    DELTAQ_updated.Total_Angular_Power = Total_Angular_Power_DELTAQ;

    if options.calculate_work
        DELTAQ_updated.LH_Linear_Work = LH_Linear_Work_DELTAQ;
        DELTAQ_updated.RH_Linear_Work = RH_Linear_Work_DELTAQ;
        DELTAQ_updated.Total_Linear_Work = Total_Linear_Work_DELTAQ;

        DELTAQ_updated.LH_Angular_Work = LH_Angular_Work_DELTAQ;
        DELTAQ_updated.RH_Angular_Work = RH_Angular_Work_DELTAQ;
        DELTAQ_updated.Total_Angular_Work = Total_Angular_Work_DELTAQ;
    end

    DELTAQ_updated.Total_Angular_Impulse_X = Total_Angular_Impulse_DELTAQ(:, 1);
    DELTAQ_updated.Total_Angular_Impulse_Y = Total_Angular_Impulse_DELTAQ(:, 2);
    DELTAQ_updated.Total_Angular_Impulse_Z = Total_Angular_Impulse_DELTAQ(:, 3);

catch ME
    warning('Error processing DELTAQ data: %s', ME.message);
    % Return the original tables as is if columns are missing
    ZTCFQ_updated = ZTCFQ;
    DELTAQ_updated = DELTAQ;
    return;
end

fprintf('ZTCFQ and DELTAQ power, work (optional), and angular impulse calculations complete.\n');
fprintf('Work calculations: %s\n', mat2str(options.calculate_work));
fprintf('Applied torques included in angular impulse: %s\n', mat2str(options.include_applied_torques));
fprintf('Force moments included in angular impulse: %s\n', mat2str(options.include_force_moments));

end
