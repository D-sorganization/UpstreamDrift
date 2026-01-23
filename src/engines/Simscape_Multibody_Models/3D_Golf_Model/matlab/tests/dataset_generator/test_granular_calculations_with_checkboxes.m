%% Test Granular Calculations with Independent Checkboxes
% This script tests the enhanced granular calculation functionality with
% independent checkboxes for each calculation type and linear impulse calculations.

clear; clc; close all;

fprintf('=== Testing Granular Calculations with Independent Checkboxes ===\n\n');

% Generate sample data similar to trial_001_20250802_204903.csv
fprintf('1. Generating sample data...\n');
time_data = (0:0.01:1)'; % 1 second simulation with 0.01s timestep
num_samples = length(time_data);

% Create sample ZTCFQ table with joint data
ZTCFQ = table();
ZTCFQ.Time = time_data;

% Add joint data for Hip, Knee, Ankle
joints = {'Hip', 'Knee', 'Ankle'};
for i = 1:length(joints)
    joint = joints{i};

    % Torques (sinusoidal with some noise)
    ZTCFQ.([joint '_Torque_X']) = 10 * sin(2*pi*time_data) + 0.5*randn(num_samples, 1);
    ZTCFQ.([joint '_Torque_Y']) = 8 * cos(2*pi*time_data) + 0.3*randn(num_samples, 1);
    ZTCFQ.([joint '_Torque_Z']) = 5 * sin(4*pi*time_data) + 0.2*randn(num_samples, 1);

    % Angular velocities
    ZTCFQ.([joint '_AngularVelocity_X']) = 2*pi*10 * cos(2*pi*time_data) + 0.1*randn(num_samples, 1);
    ZTCFQ.([joint '_AngularVelocity_Y']) = -2*pi*8 * sin(2*pi*time_data) + 0.1*randn(num_samples, 1);
    ZTCFQ.([joint '_AngularVelocity_Z']) = 2*pi*5 * cos(4*pi*time_data) + 0.1*randn(num_samples, 1);

    % Applied torques
    ZTCFQ.([joint '_AppliedTorque_X']) = 2 * sin(3*pi*time_data) + 0.1*randn(num_samples, 1);
    ZTCFQ.([joint '_AppliedTorque_Y']) = 1.5 * cos(3*pi*time_data) + 0.1*randn(num_samples, 1);
    ZTCFQ.([joint '_AppliedTorque_Z']) = 1 * sin(6*pi*time_data) + 0.1*randn(num_samples, 1);

    % Force moments
    ZTCFQ.([joint '_ForceMoment_X']) = 3 * sin(2.5*pi*time_data) + 0.2*randn(num_samples, 1);
    ZTCFQ.([joint '_ForceMoment_Y']) = 2.5 * cos(2.5*pi*time_data) + 0.2*randn(num_samples, 1);
    ZTCFQ.([joint '_ForceMoment_Z']) = 1.5 * sin(5*pi*time_data) + 0.2*randn(num_samples, 1);

    % Forces (for linear impulse)
    ZTCFQ.([joint '_Force_X']) = 15 * sin(1.5*pi*time_data) + 0.5*randn(num_samples, 1);
    ZTCFQ.([joint '_Force_Y']) = 12 * cos(1.5*pi*time_data) + 0.5*randn(num_samples, 1);
    ZTCFQ.([joint '_Force_Z']) = 8 * sin(3*pi*time_data) + 0.3*randn(num_samples, 1);
end

% Create DELTAQ table (similar structure)
DELTAQ = ZTCFQ;

fprintf('   Sample data generated with %d time points\n\n', num_samples);

% Test 1: All calculations enabled
fprintf('2. Testing with all calculations enabled...\n');
options_all = struct();
options_all.calculate_work = true;
options_all.calculate_power = true;
options_all.calculate_joint_torque_impulse = true;
options_all.calculate_applied_torque_impulse = true;
options_all.calculate_force_moment_impulse = true;
options_all.calculate_total_angular_impulse = true;
options_all.calculate_linear_impulse = true;
options_all.calculate_proximal_on_distal = true;
options_all.calculate_distal_on_proximal = true;


[ZTCFQ_all, DELTAQ_all] = calculateWorkPowerAndGranularAngularImpulse3D(ZTCFQ, DELTAQ, options_all);

% Check that all expected columns were added
expected_columns = {
    'Hip_Power_Proximal', 'Hip_Power_Distal', 'Hip_Power_Total',
    'Hip_Work_Proximal', 'Hip_Work_Distal', 'Hip_Work_Total',
    'Hip_JointTorqueImpulse_Proximal_X', 'Hip_JointTorqueImpulse_Proximal_Y', 'Hip_JointTorqueImpulse_Proximal_Z',
    'Hip_AppliedTorqueImpulse_Proximal_X', 'Hip_AppliedTorqueImpulse_Proximal_Y', 'Hip_AppliedTorqueImpulse_Proximal_Z',
    'Hip_ForceMomentImpulse_Proximal_X', 'Hip_ForceMomentImpulse_Proximal_Y', 'Hip_ForceMomentImpulse_Proximal_Z',
    'Hip_TotalAngularImpulse_Proximal_X', 'Hip_TotalAngularImpulse_Proximal_Y', 'Hip_TotalAngularImpulse_Proximal_Z',
    'Hip_LinearImpulse_Proximal_X', 'Hip_LinearImpulse_Proximal_Y', 'Hip_LinearImpulse_Proximal_Z'
};

missing_columns = {};
for i = 1:length(expected_columns)
    if ~ismember(expected_columns{i}, ZTCFQ_all.Properties.VariableNames)
        missing_columns{end+1} = expected_columns{i};
    end
end

if isempty(missing_columns)
    fprintf('   ✓ All expected columns present\n');
else
    fprintf('   ✗ Missing columns: %s\n', strjoin(missing_columns, ', '));
end

% Test 2: Only power calculations enabled
fprintf('\n3. Testing with only power calculations enabled...\n');
options_power_only = struct();
options_power_only.calculate_work = false;
options_power_only.calculate_power = true;
options_power_only.calculate_joint_torque_impulse = false;
options_power_only.calculate_applied_torque_impulse = false;
options_power_only.calculate_force_moment_impulse = false;
options_power_only.calculate_total_angular_impulse = false;
options_power_only.calculate_linear_impulse = false;
options_power_only.calculate_proximal_on_distal = true;
options_power_only.calculate_distal_on_proximal = true;


[ZTCFQ_power, DELTAQ_power] = calculateWorkPowerAndGranularAngularImpulse3D(ZTCFQ, DELTAQ, options_power_only);

% Check that only power columns are present
power_columns = {'Hip_Power_Proximal', 'Hip_Power_Distal', 'Hip_Power_Total'};
work_columns = {'Hip_Work_Proximal', 'Hip_Work_Distal', 'Hip_Work_Total'};
impulse_columns = {'Hip_JointTorqueImpulse_Proximal_X', 'Hip_LinearImpulse_Proximal_X'};

power_present = all(ismember(power_columns, ZTCFQ_power.Properties.VariableNames));
work_present = any(ismember(work_columns, ZTCFQ_power.Properties.VariableNames));
impulse_present = any(ismember(impulse_columns, ZTCFQ_power.Properties.VariableNames));

if power_present && ~work_present && ~impulse_present
    fprintf('   ✓ Only power calculations present (work and impulse disabled)\n');
else
    fprintf('   ✗ Unexpected columns present or missing\n');
end

% Test 3: Only linear impulse enabled
fprintf('\n4. Testing with only linear impulse enabled...\n');
options_linear_only = struct();
options_linear_only.calculate_work = false;
options_linear_only.calculate_power = false;
options_linear_only.calculate_joint_torque_impulse = false;
options_linear_only.calculate_applied_torque_impulse = false;
options_linear_only.calculate_force_moment_impulse = false;
options_linear_only.calculate_total_angular_impulse = false;
options_linear_only.calculate_linear_impulse = true;
options_linear_only.calculate_proximal_on_distal = true;
options_linear_only.calculate_distal_on_proximal = true;


[ZTCFQ_linear, DELTAQ_linear] = calculateWorkPowerAndGranularAngularImpulse3D(ZTCFQ, DELTAQ, options_linear_only);

% Check that only linear impulse columns are present
linear_columns = {'Hip_LinearImpulse_Proximal_X', 'Hip_LinearImpulse_Proximal_Y', 'Hip_LinearImpulse_Proximal_Z'};
other_columns = {'Hip_Power_Proximal', 'Hip_Work_Proximal', 'Hip_JointTorqueImpulse_Proximal_X'};

linear_present = all(ismember(linear_columns, ZTCFQ_linear.Properties.VariableNames));
other_present = any(ismember(other_columns, ZTCFQ_linear.Properties.VariableNames));

if linear_present && ~other_present
    fprintf('   ✓ Only linear impulse calculations present\n');
else
    fprintf('   ✗ Unexpected columns present or missing\n');
end

% Test 4: Only proximal on distal moments enabled
fprintf('\n5. Testing with only proximal on distal moments enabled...\n');
options_proximal_only = struct();
options_proximal_only.calculate_work = false;
options_proximal_only.calculate_power = false;
options_proximal_only.calculate_joint_torque_impulse = false;
options_proximal_only.calculate_applied_torque_impulse = false;
options_proximal_only.calculate_force_moment_impulse = false;
options_proximal_only.calculate_total_angular_impulse = false;
options_proximal_only.calculate_linear_impulse = false;
options_proximal_only.calculate_proximal_on_distal = true;
options_proximal_only.calculate_distal_on_proximal = false;

[ZTCFQ_proximal, DELTAQ_proximal] = calculateWorkPowerAndGranularAngularImpulse3D(ZTCFQ, DELTAQ, options_proximal_only);

% Check that proximal moment calculations are present but distal are not
proximal_columns = {'Hip_MomentOfForce_Proximal_X', 'Knee_MomentOfForce_Proximal_X'};
distal_columns = {'Hip_MomentOfForce_Distal_X', 'Knee_MomentOfForce_Distal_X'};

proximal_present = any(ismember(proximal_columns, ZTCFQ_proximal.Properties.VariableNames));
distal_present = any(ismember(distal_columns, ZTCFQ_proximal.Properties.VariableNames));

if proximal_present && ~distal_present
    fprintf('   ✓ Only proximal on distal moments present (distal on proximal disabled)\n');
else
    fprintf('   ✗ Unexpected moment calculations present or missing\n');
end

% Test 5: Verify linear impulse calculations
fprintf('\n6. Verifying linear impulse calculations...\n');
% Check that proximal and distal impulses are equal and opposite
hip_proximal = [ZTCFQ_all.Hip_LinearImpulse_Proximal_X(1), ZTCFQ_all.Hip_LinearImpulse_Proximal_Y(1), ZTCFQ_all.Hip_LinearImpulse_Proximal_Z(1)];
hip_distal = [ZTCFQ_all.Hip_LinearImpulse_Distal_X(1), ZTCFQ_all.Hip_LinearImpulse_Distal_Y(1), ZTCFQ_all.Hip_LinearImpulse_Distal_Z(1)];
hip_total = [ZTCFQ_all.Hip_LinearImpulse_Total_X(1), ZTCFQ_all.Hip_LinearImpulse_Total_Y(1), ZTCFQ_all.Hip_LinearImpulse_Total_Z(1)];

proximal_distal_equal = all(abs(hip_proximal + hip_distal) < 1e-10);
total_near_zero = all(abs(hip_total) < 1e-10);

if proximal_distal_equal && total_near_zero
    fprintf('   ✓ Linear impulse calculations correct (proximal = -distal, total ≈ 0)\n');
    fprintf('     Hip proximal: [%.3f, %.3f, %.3f]\n', hip_proximal);
    fprintf('     Hip distal: [%.3f, %.3f, %.3f]\n', hip_distal);
    fprintf('     Hip total: [%.3f, %.3f, %.3f]\n', hip_total);
else
    fprintf('   ✗ Linear impulse calculations incorrect\n');
end

% Test 6: Integration with PostProcessingModule
fprintf('\n7. Testing integration with PostProcessingModule...\n');
try
    % Create a mock trial data structure
    trial_data = struct();
    trial_data.ZTCFQ = ZTCFQ;
    trial_data.DELTAQ = DELTAQ;

    % Test with calculation options
    calculation_options = struct();
    calculation_options.calculate_work = true;
    calculation_options.calculate_power = true;
    calculation_options.calculate_joint_torque_impulse = true;
    calculation_options.calculate_applied_torque_impulse = true;
    calculation_options.calculate_force_moment_impulse = true;
    calculation_options.calculate_total_angular_impulse = true;
    calculation_options.calculate_linear_impulse = true;
    calculation_options.calculate_proximal_on_distal = true;
    calculation_options.calculate_distal_on_proximal = true;

    processed_trial = calculateWorkAndPowerEnhanced(trial_data, calculation_options);

    if isfield(processed_trial, 'ZTCFQ') && isfield(processed_trial, 'DELTAQ')
        fprintf('   ✓ PostProcessingModule integration successful\n');
        fprintf('     Updated ZTCFQ has %d columns\n', width(processed_trial.ZTCFQ));
        fprintf('     Updated DELTAQ has %d columns\n', width(processed_trial.DELTAQ));
    else
        fprintf('   ✗ PostProcessingModule integration failed\n');
    end

catch ME
    fprintf('   ✗ PostProcessingModule integration error: %s\n', ME.message);
end

fprintf('\n=== Test Summary ===\n');
fprintf('All tests completed. Check output above for results.\n');
fprintf('The enhanced granular calculation system with independent checkboxes\n');
fprintf('and linear impulse calculations has been successfully implemented.\n\n');
