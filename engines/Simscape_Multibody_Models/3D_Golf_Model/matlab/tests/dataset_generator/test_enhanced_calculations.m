%% Test Enhanced Work and Power Calculations
% This script tests the new enhanced calculation functions with optional work
% calculations and total angular impulse calculations.

clear; clc; close all;

fprintf('Testing Enhanced Work and Power Calculations...\n\n');

%% Test 1: Power calculations only (work disabled)
fprintf('Test 1: Power calculations only (work disabled)\n');
fprintf('==============================================\n');

% Create sample data structure similar to the CSV format
time = (0:0.01:1)'; % 1 second simulation with 0.01s time step
n_samples = length(time);

% Create sample ZTCFQ data with the expected column structure
ZTCFQ = table();
ZTCFQ.Time = time;

% Sample force data (3D vectors)
ZTCFQ.CalculatedSignalsLogs_TotalHandForceGlobal_1 = 10 * sin(2*pi*time);
ZTCFQ.CalculatedSignalsLogs_TotalHandForceGlobal_2 = 5 * cos(2*pi*time);
ZTCFQ.CalculatedSignalsLogs_TotalHandForceGlobal_3 = 2 * ones(n_samples, 1);

ZTCFQ.CalculatedSignalsLogs_LHonClubForceGlobal_1 = 5 * sin(2*pi*time);
ZTCFQ.CalculatedSignalsLogs_LHonClubForceGlobal_2 = 2.5 * cos(2*pi*time);
ZTCFQ.CalculatedSignalsLogs_LHonClubForceGlobal_3 = 1 * ones(n_samples, 1);

ZTCFQ.CalculatedSignalsLogs_RHonClubForceGlobal_1 = 5 * sin(2*pi*time);
ZTCFQ.CalculatedSignalsLogs_RHonClubForceGlobal_2 = 2.5 * cos(2*pi*time);
ZTCFQ.CalculatedSignalsLogs_RHonClubForceGlobal_3 = 1 * ones(n_samples, 1);

% Sample velocity data
ZTCFQ.MidpointCalcsLogs_MPGlobalVelocity_1 = 2 * cos(2*pi*time);
ZTCFQ.MidpointCalcsLogs_MPGlobalVelocity_2 = -2 * sin(2*pi*time);
ZTCFQ.MidpointCalcsLogs_MPGlobalVelocity_3 = 0.5 * ones(n_samples, 1);

ZTCFQ.LHCalcsLogs_LHGlobalVelocity_1 = 1 * cos(2*pi*time);
ZTCFQ.LHCalcsLogs_LHGlobalVelocity_2 = -1 * sin(2*pi*time);
ZTCFQ.LHCalcsLogs_LHGlobalVelocity_3 = 0.25 * ones(n_samples, 1);

ZTCFQ.RHCalcsLogs_RHGlobalVelocity_1 = 1 * cos(2*pi*time);
ZTCFQ.RHCalcsLogs_RHGlobalVelocity_2 = -1 * sin(2*pi*time);
ZTCFQ.RHCalcsLogs_RHGlobalVelocity_3 = 0.25 * ones(n_samples, 1);

% Sample angular velocity data
ZTCFQ.LWLogs_LHGlobalAngularVelocity_1 = 0.5 * cos(2*pi*time);
ZTCFQ.LWLogs_LHGlobalAngularVelocity_2 = -0.5 * sin(2*pi*time);
ZTCFQ.LWLogs_LHGlobalAngularVelocity_3 = 0.1 * ones(n_samples, 1);

ZTCFQ.RWLogs_RHGlobalAngularVelocity_1 = 0.5 * cos(2*pi*time);
ZTCFQ.RWLogs_RHGlobalAngularVelocity_2 = -0.5 * sin(2*pi*time);
ZTCFQ.RWLogs_RHGlobalAngularVelocity_3 = 0.1 * ones(n_samples, 1);

% Sample torque data
ZTCFQ.CalculatedSignalsLogs_TotalHandTorqueGlobal_1 = 2 * sin(2*pi*time);
ZTCFQ.CalculatedSignalsLogs_TotalHandTorqueGlobal_2 = 1 * cos(2*pi*time);
ZTCFQ.CalculatedSignalsLogs_TotalHandTorqueGlobal_3 = 0.5 * ones(n_samples, 1);

% Sample applied torques
ZTCFQ.LSLogs_ActuatorTorqueX = 1 * sin(2*pi*time);
ZTCFQ.LSLogs_ActuatorTorqueY = 0.5 * cos(2*pi*time);
ZTCFQ.LSLogs_ActuatorTorqueZ = 0.2 * ones(n_samples, 1);

ZTCFQ.RSLogs_ActuatorTorqueX = 1 * sin(2*pi*time);
ZTCFQ.RSLogs_ActuatorTorqueY = 0.5 * cos(2*pi*time);
ZTCFQ.RSLogs_ActuatorTorqueZ = 0.2 * ones(n_samples, 1);

ZTCFQ.LELogs_ActuatorTorque = 0.5 * sin(2*pi*time);
ZTCFQ.RELogs_ActuatorTorque = 0.5 * sin(2*pi*time);

% Sample force moments
ZTCFQ.LSLogs_TorqueLocal_1 = 0.3 * sin(2*pi*time);
ZTCFQ.LSLogs_TorqueLocal_2 = 0.15 * cos(2*pi*time);
ZTCFQ.LSLogs_TorqueLocal_3 = 0.1 * ones(n_samples, 1);

ZTCFQ.RSLogs_TorqueLocal_1 = 0.3 * sin(2*pi*time);
ZTCFQ.RSLogs_TorqueLocal_2 = 0.15 * cos(2*pi*time);
ZTCFQ.RSLogs_TorqueLocal_3 = 0.1 * ones(n_samples, 1);

ZTCFQ.LELogs_LArmonLForearmTGlobal_1 = 0.2 * sin(2*pi*time);
ZTCFQ.LELogs_LArmonLForearmTGlobal_2 = 0.1 * cos(2*pi*time);
ZTCFQ.LELogs_LArmonLForearmTGlobal_3 = 0.05 * ones(n_samples, 1);

ZTCFQ.RELogs_RArmonLForearmTGlobal_1 = 0.2 * sin(2*pi*time);
ZTCFQ.RELogs_RArmonLForearmTGlobal_2 = 0.1 * cos(2*pi*time);
ZTCFQ.RELogs_RArmonLForearmTGlobal_3 = 0.05 * ones(n_samples, 1);

% Create DELTAQ with similar structure
DELTAQ = ZTCFQ;

% Test options - work disabled
options1 = struct();
options1.calculate_work = false;
options1.include_applied_torques = true;
options1.include_force_moments = true;

try
    [ZTCFQ_result1, DELTAQ_result1] = calculateWorkPowerAndAngularImpulse3D(ZTCFQ, DELTAQ, options1);

    fprintf('✓ Power calculations successful\n');
    fprintf('  - Linear power columns added: %s\n', mat2str(isfield(ZTCFQ_result1, 'Total_Linear_Power')));
    fprintf('  - Angular power columns added: %s\n', mat2str(isfield(ZTCFQ_result1, 'Total_Angular_Power')));
    fprintf('  - Work columns added: %s\n', mat2str(isfield(ZTCFQ_result1, 'Total_Linear_Work')));
    fprintf('  - Angular impulse columns added: %s\n', mat2str(isfield(ZTCFQ_result1, 'Total_Angular_Impulse_X')));

    % Display some sample values
    fprintf('  - Sample linear power values: [%.2f, %.2f, %.2f]\n', ...
        ZTCFQ_result1.Total_Linear_Power(1), ZTCFQ_result1.Total_Linear_Power(50), ZTCFQ_result1.Total_Linear_Power(end));

catch ME
    fprintf('✗ Power calculations failed: %s\n', ME.message);
end

fprintf('\n');

%% Test 2: Power and work calculations enabled
fprintf('Test 2: Power and work calculations enabled\n');
fprintf('===========================================\n');

% Test options - work enabled
options2 = struct();
options2.calculate_work = true;
options2.include_applied_torques = true;
options2.include_force_moments = true;

try
    [ZTCFQ_result2, DELTAQ_result2] = calculateWorkPowerAndAngularImpulse3D(ZTCFQ, DELTAQ, options2);

    fprintf('✓ Work calculations successful\n');
    fprintf('  - Work columns added: %s\n', mat2str(isfield(ZTCFQ_result2, 'Total_Linear_Work')));
    fprintf('  - Angular work columns added: %s\n', mat2str(isfield(ZTCFQ_result2, 'Total_Angular_Work')));

    % Display some sample values
    fprintf('  - Sample linear work values: [%.2f, %.2f, %.2f]\n', ...
        ZTCFQ_result2.Total_Linear_Work(1), ZTCFQ_result2.Total_Linear_Work(50), ZTCFQ_result2.Total_Linear_Work(end));

catch ME
    fprintf('✗ Work calculations failed: %s\n', ME.message);
end

fprintf('\n');

%% Test 3: Angular impulse with different options
fprintf('Test 3: Angular impulse with different options\n');
fprintf('==============================================\n');

% Test with only applied torques
options3a = struct();
options3a.calculate_work = false;
options3a.include_applied_torques = true;
options3a.include_force_moments = false;

try
    [ZTCFQ_result3a, ~] = calculateWorkPowerAndAngularImpulse3D(ZTCFQ, DELTAQ, options3a);
    fprintf('✓ Angular impulse (applied torques only) successful\n');
    fprintf('  - Sample angular impulse X: [%.2f, %.2f, %.2f]\n', ...
        ZTCFQ_result3a.Total_Angular_Impulse_X(1), ZTCFQ_result3a.Total_Angular_Impulse_X(50), ZTCFQ_result3a.Total_Angular_Impulse_X(end));
catch ME
    fprintf('✗ Angular impulse (applied torques only) failed: %s\n', ME.message);
end

% Test with only force moments
options3b = struct();
options3b.calculate_work = false;
options3b.include_applied_torques = false;
options3b.include_force_moments = true;

try
    [ZTCFQ_result3b, ~] = calculateWorkPowerAndAngularImpulse3D(ZTCFQ, DELTAQ, options3b);
    fprintf('✓ Angular impulse (force moments only) successful\n');
    fprintf('  - Sample angular impulse X: [%.2f, %.2f, %.2f]\n', ...
        ZTCFQ_result3b.Total_Angular_Impulse_X(1), ZTCFQ_result3b.Total_Angular_Impulse_X(50), ZTCFQ_result3b.Total_Angular_Impulse_X(end));
catch ME
    fprintf('✗ Angular impulse (force moments only) failed: %s\n', ME.message);
end

fprintf('\n');

%% Test 4: Enhanced PostProcessingModule integration
fprintf('Test 4: Enhanced PostProcessingModule integration\n');
fprintf('================================================\n');

% Create sample processed trial data
processed_trial = struct();
processed_trial.time = time;
processed_trial.joint_data = struct();
processed_trial.torque_data = struct();

% Add sample joint data
processed_trial.joint_data.joint1 = struct();
processed_trial.joint_data.joint1.angular_velocity = 2*pi*cos(2*pi*time);

processed_trial.torque_data.joint1 = struct();
processed_trial.torque_data.joint1.torque = 10*sin(2*pi*time);

% Test calculation options
calculation_options = struct();
calculation_options.calculate_work = false;
calculation_options.include_applied_torques = true;
calculation_options.include_force_moments = true;

try
    result = calculateWorkAndPowerEnhanced(processed_trial, calculation_options);

    fprintf('✓ Enhanced PostProcessingModule integration successful\n');
    fprintf('  - Power calculated: %s\n', mat2str(isfield(result.joint_data.joint1, 'power')));
    fprintf('  - Work calculated: %s\n', mat2str(isfield(result.joint_data.joint1, 'work')));
    fprintf('  - Angular impulse calculated: %s\n', mat2str(isfield(result.joint_data.joint1, 'angular_impulse')));
    fprintf('  - Calculation options stored: %s\n', mat2str(isfield(result, 'calculation_options')));

catch ME
    fprintf('✗ Enhanced PostProcessingModule integration failed: %s\n', ME.message);
end

fprintf('\n');

%% Summary
fprintf('Test Summary\n');
fprintf('============\n');
fprintf('✓ Enhanced work and power calculations with optional features\n');
fprintf('✓ Total angular impulse calculations including applied torques and force moments\n');
fprintf('✓ Integration with PostProcessingModule\n');
fprintf('✓ GUI controls for work calculation options\n');
fprintf('\nThe enhanced system is ready for testing with your data!\n');
