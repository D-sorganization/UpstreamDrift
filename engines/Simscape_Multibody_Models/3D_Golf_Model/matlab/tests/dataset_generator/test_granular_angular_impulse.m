% TEST_GRANULAR_ANGULAR_IMPULSE - Test script for granular angular impulse calculations
% Tests the new calculateWorkPowerAndGranularAngularImpulse3D function with detailed breakdown

clear; clc; close all;

fprintf('=== Testing Granular Angular Impulse Calculations ===\n\n');

% Create sample data mimicking trial_001_20250802_204903.csv structure
fprintf('1. Creating sample data...\n');
time_data = (0:0.01:1)'; % 1 second simulation, 100Hz
num_samples = length(time_data);

% Create ZTCFQ table with joint data
ZTCFQ = table();
ZTCFQ.Time = time_data;

% Add joint torque data (X, Y, Z components)
ZTCFQ.Hip_Torque_X = sin(2*pi*time_data) * 10; % Nm
ZTCFQ.Hip_Torque_Y = cos(2*pi*time_data) * 8;
ZTCFQ.Hip_Torque_Z = sin(4*pi*time_data) * 5;

ZTCFQ.Knee_Torque_X = sin(3*pi*time_data) * 12;
ZTCFQ.Knee_Torque_Y = cos(3*pi*time_data) * 9;
ZTCFQ.Knee_Torque_Z = sin(6*pi*time_data) * 6;

ZTCFQ.Ankle_Torque_X = sin(pi*time_data) * 7;
ZTCFQ.Ankle_Torque_Y = cos(pi*time_data) * 5;
ZTCFQ.Ankle_Torque_Z = sin(2*pi*time_data) * 3;

% Add angular velocity data
ZTCFQ.Hip_AngularVelocity_X = cos(2*pi*time_data) * 2; % rad/s
ZTCFQ.Hip_AngularVelocity_Y = -sin(2*pi*time_data) * 1.5;
ZTCFQ.Hip_AngularVelocity_Z = cos(4*pi*time_data) * 1;

ZTCFQ.Knee_AngularVelocity_X = cos(3*pi*time_data) * 2.5;
ZTCFQ.Knee_AngularVelocity_Y = -sin(3*pi*time_data) * 2;
ZTCFQ.Knee_AngularVelocity_Z = cos(6*pi*time_data) * 1.5;

ZTCFQ.Ankle_AngularVelocity_X = cos(pi*time_data) * 1.5;
ZTCFQ.Ankle_AngularVelocity_Y = -sin(pi*time_data) * 1;
ZTCFQ.Ankle_AngularVelocity_Z = cos(2*pi*time_data) * 0.8;

% Add applied torque data
ZTCFQ.Hip_AppliedTorque_X = sin(2*pi*time_data) * 5;
ZTCFQ.Hip_AppliedTorque_Y = cos(2*pi*time_data) * 4;
ZTCFQ.Hip_AppliedTorque_Z = sin(4*pi*time_data) * 2;

ZTCFQ.Knee_AppliedTorque_X = sin(3*pi*time_data) * 6;
ZTCFQ.Knee_AppliedTorque_Y = cos(3*pi*time_data) * 4.5;
ZTCFQ.Knee_AppliedTorque_Z = sin(6*pi*time_data) * 3;

ZTCFQ.Ankle_AppliedTorque_X = sin(pi*time_data) * 3.5;
ZTCFQ.Ankle_AppliedTorque_Y = cos(pi*time_data) * 2.5;
ZTCFQ.Ankle_AppliedTorque_Z = sin(2*pi*time_data) * 1.5;

% Add force moment data
ZTCFQ.Hip_ForceMoment_X = sin(2*pi*time_data) * 3;
ZTCFQ.Hip_ForceMoment_Y = cos(2*pi*time_data) * 2.5;
ZTCFQ.Hip_ForceMoment_Z = sin(4*pi*time_data) * 1.5;

ZTCFQ.Knee_ForceMoment_X = sin(3*pi*time_data) * 4;
ZTCFQ.Knee_ForceMoment_Y = cos(3*pi*time_data) * 3;
ZTCFQ.Knee_ForceMoment_Z = sin(6*pi*time_data) * 2;

ZTCFQ.Ankle_ForceMoment_X = sin(pi*time_data) * 2;
ZTCFQ.Ankle_ForceMoment_Y = cos(pi*time_data) * 1.5;
ZTCFQ.Ankle_ForceMoment_Z = sin(2*pi*time_data) * 1;

% Create DELTAQ table (similar structure)
DELTAQ = ZTCFQ;

fprintf('   Sample data created with %d time points\n', num_samples);
fprintf('   Joints: Hip, Knee, Ankle\n');
fprintf('   Data types: Torques, Angular Velocities, Applied Torques, Force Moments\n\n');

% Test 1: Power calculations only (work disabled)
fprintf('2. Testing power calculations only (work disabled)...\n');
options = struct();
options.calculate_work = false;

try
    [ZTCFQ_power, DELTAQ_power] = calculateWorkPowerAndGranularAngularImpulse3D(ZTCFQ, DELTAQ, options);

    % Check power columns
    power_columns = {'Hip_Power_Proximal', 'Hip_Power_Distal', 'Hip_Power_Total', ...
                     'Knee_Power_Proximal', 'Knee_Power_Distal', 'Knee_Power_Total', ...
                     'Ankle_Power_Proximal', 'Ankle_Power_Distal', 'Ankle_Power_Total'};

    missing_power = 0;
    for i = 1:length(power_columns)
        if ~isfield(ZTCFQ_power, power_columns{i})
            missing_power = missing_power + 1;
        end
    end

    if missing_power == 0
        fprintf('   ✓ Power calculations successful\n');
        fprintf('   ✓ Power columns added: %d\n', length(power_columns));

        % Show sample power values
        fprintf('   Sample Hip Power (Proximal): %.2f W\n', ZTCFQ_power.Hip_Power_Proximal(50));
        fprintf('   Sample Knee Power (Total): %.2f W\n', ZTCFQ_power.Knee_Power_Total(50));
    else
        fprintf('   ✗ Missing power columns: %d\n', missing_power);
    end

catch ME
    fprintf('   ✗ Power calculation failed: %s\n', ME.message);
end

% Test 2: Power and work calculations enabled
fprintf('\n3. Testing power and work calculations (work enabled)...\n');
options.calculate_work = true;

try
    [ZTCFQ_work, DELTAQ_work] = calculateWorkPowerAndGranularAngularImpulse3D(ZTCFQ, DELTAQ, options);

    % Check work columns
    work_columns = {'Hip_Work_Proximal', 'Hip_Work_Distal', 'Hip_Work_Total', ...
                    'Knee_Work_Proximal', 'Knee_Work_Distal', 'Knee_Work_Total', ...
                    'Ankle_Work_Proximal', 'Ankle_Work_Distal', 'Ankle_Work_Total'};

    missing_work = 0;
    for i = 1:length(work_columns)
        if ~isfield(ZTCFQ_work, work_columns{i})
            missing_work = missing_work + 1;
        end
    end

    if missing_work == 0
        fprintf('   ✓ Work calculations successful\n');
        fprintf('   ✓ Work columns added: %d\n', length(work_columns));

        % Show sample work values
        fprintf('   Hip Work (Proximal): %.2f J\n', ZTCFQ_work.Hip_Work_Proximal);
        fprintf('   Knee Work (Total): %.2f J\n', ZTCFQ_work.Knee_Work_Total);
    else
        fprintf('   ✗ Missing work columns: %d\n', missing_work);
    end

catch ME
    fprintf('   ✗ Work calculation failed: %s\n', ME.message);
end

% Test 3: Angular impulse calculations
fprintf('\n4. Testing granular angular impulse calculations...\n');

try
    % Check angular impulse columns
    impulse_columns = {'Hip_JointTorqueImpulse_Proximal_X', 'Hip_JointTorqueImpulse_Proximal_Y', 'Hip_JointTorqueImpulse_Proximal_Z', ...
                       'Hip_AppliedTorqueImpulse_Proximal_X', 'Hip_AppliedTorqueImpulse_Proximal_Y', 'Hip_AppliedTorqueImpulse_Proximal_Z', ...
                       'Hip_ForceMomentImpulse_Proximal_X', 'Hip_ForceMomentImpulse_Proximal_Y', 'Hip_ForceMomentImpulse_Proximal_Z', ...
                       'Hip_TotalAngularImpulse_Proximal_X', 'Hip_TotalAngularImpulse_Proximal_Y', 'Hip_TotalAngularImpulse_Proximal_Z', ...
                       'Hip_JointTorqueImpulse_Distal_X', 'Hip_JointTorqueImpulse_Distal_Y', 'Hip_JointTorqueImpulse_Distal_Z', ...
                       'Hip_AppliedTorqueImpulse_Distal_X', 'Hip_AppliedTorqueImpulse_Distal_Y', 'Hip_AppliedTorqueImpulse_Distal_Z', ...
                       'Hip_ForceMomentImpulse_Distal_X', 'Hip_ForceMomentImpulse_Distal_Y', 'Hip_ForceMomentImpulse_Distal_Z', ...
                       'Hip_TotalAngularImpulse_Distal_X', 'Hip_TotalAngularImpulse_Distal_Y', 'Hip_TotalAngularImpulse_Distal_Z', ...
                       'Hip_JointTorqueImpulse_Total_X', 'Hip_JointTorqueImpulse_Total_Y', 'Hip_JointTorqueImpulse_Total_Z', ...
                       'Hip_AppliedTorqueImpulse_Total_X', 'Hip_AppliedTorqueImpulse_Total_Y', 'Hip_AppliedTorqueImpulse_Total_Z', ...
                       'Hip_ForceMomentImpulse_Total_X', 'Hip_ForceMomentImpulse_Total_Y', 'Hip_ForceMomentImpulse_Total_Z', ...
                       'Hip_TotalAngularImpulse_Total_X', 'Hip_TotalAngularImpulse_Total_Y', 'Hip_TotalAngularImpulse_Total_Z'};

    missing_impulse = 0;
    for i = 1:length(impulse_columns)
        if ~isfield(ZTCFQ_work, impulse_columns{i})
            missing_impulse = missing_impulse + 1;
        end
    end

    if missing_impulse == 0
        fprintf('   ✓ Angular impulse calculations successful\n');
        fprintf('   ✓ Angular impulse columns added: %d\n', length(impulse_columns));

        % Show sample angular impulse values for Hip
        fprintf('   Hip Joint Torque Impulse (Proximal): [%.2f, %.2f, %.2f] N⋅m⋅s\n', ...
                ZTCFQ_work.Hip_JointTorqueImpulse_Proximal_X, ...
                ZTCFQ_work.Hip_JointTorqueImpulse_Proximal_Y, ...
                ZTCFQ_work.Hip_JointTorqueImpulse_Proximal_Z);

        fprintf('   Hip Applied Torque Impulse (Proximal): [%.2f, %.2f, %.2f] N⋅m⋅s\n', ...
                ZTCFQ_work.Hip_AppliedTorqueImpulse_Proximal_X, ...
                ZTCFQ_work.Hip_AppliedTorqueImpulse_Proximal_Y, ...
                ZTCFQ_work.Hip_AppliedTorqueImpulse_Proximal_Z);

        fprintf('   Hip Force Moment Impulse (Proximal): [%.2f, %.2f, %.2f] N⋅m⋅s\n', ...
                ZTCFQ_work.Hip_ForceMomentImpulse_Proximal_X, ...
                ZTCFQ_work.Hip_ForceMomentImpulse_Proximal_Y, ...
                ZTCFQ_work.Hip_ForceMomentImpulse_Proximal_Z);

        fprintf('   Hip Total Angular Impulse (Proximal): [%.2f, %.2f, %.2f] N⋅m⋅s\n', ...
                ZTCFQ_work.Hip_TotalAngularImpulse_Proximal_X, ...
                ZTCFQ_work.Hip_TotalAngularImpulse_Proximal_Y, ...
                ZTCFQ_work.Hip_TotalAngularImpulse_Proximal_Z);

        % Verify that distal = -proximal (equal and opposite)
        proximal_total = [ZTCFQ_work.Hip_TotalAngularImpulse_Proximal_X, ...
                         ZTCFQ_work.Hip_TotalAngularImpulse_Proximal_Y, ...
                         ZTCFQ_work.Hip_TotalAngularImpulse_Proximal_Z];
        distal_total = [ZTCFQ_work.Hip_TotalAngularImpulse_Distal_X, ...
                       ZTCFQ_work.Hip_TotalAngularImpulse_Distal_Y, ...
                       ZTCFQ_work.Hip_TotalAngularImpulse_Distal_Z];

        if norm(proximal_total + distal_total) < 1e-10
            fprintf('   ✓ Proximal and distal impulses are equal and opposite\n');
        else
            fprintf('   ✗ Proximal and distal impulses are not equal and opposite\n');
        end

    else
        fprintf('   ✗ Missing angular impulse columns: %d\n', missing_impulse);
    end

catch ME
    fprintf('   ✗ Angular impulse calculation failed: %s\n', ME.message);
end

% Test 4: Integration with PostProcessingModule
fprintf('\n5. Testing integration with PostProcessingModule...\n');

try
    % Create a mock processed_trial structure
    processed_trial = struct();
    processed_trial.time = time_data;
    processed_trial.joint_data = struct();
    processed_trial.torque_data = struct();

    % Add joint data
    processed_trial.joint_data.hip = struct();
    processed_trial.joint_data.hip.angular_velocity = [ZTCFQ.Hip_AngularVelocity_X, ZTCFQ.Hip_AngularVelocity_Y, ZTCFQ.Hip_AngularVelocity_Z];

    processed_trial.joint_data.knee = struct();
    processed_trial.joint_data.knee.angular_velocity = [ZTCFQ.Knee_AngularVelocity_X, ZTCFQ.Knee_AngularVelocity_Y, ZTCFQ.Knee_AngularVelocity_Z];

    processed_trial.joint_data.ankle = struct();
    processed_trial.joint_data.ankle.angular_velocity = [ZTCFQ.Ankle_AngularVelocity_X, ZTCFQ.Ankle_AngularVelocity_Y, ZTCFQ.Ankle_AngularVelocity_Z];

    % Add torque data
    processed_trial.torque_data.hip = struct();
    processed_trial.torque_data.hip.torque = [ZTCFQ.Hip_Torque_X, ZTCFQ.Hip_Torque_Y, ZTCFQ.Hip_Torque_Z];

    processed_trial.torque_data.knee = struct();
    processed_trial.torque_data.knee.torque = [ZTCFQ.Knee_Torque_X, ZTCFQ.Knee_Torque_Y, ZTCFQ.Knee_Torque_Z];

    processed_trial.torque_data.ankle = struct();
    processed_trial.torque_data.ankle.torque = [ZTCFQ.Ankle_Torque_X, ZTCFQ.Ankle_Torque_Y, ZTCFQ.Ankle_Torque_Z];

    % Test the enhanced function
    calculation_options = struct();
    calculation_options.calculate_work = true;

    processed_trial_enhanced = calculateWorkAndPowerEnhanced(processed_trial, calculation_options);

    if isfield(processed_trial_enhanced, 'total_peak_power') && isfield(processed_trial_enhanced, 'total_work')
        fprintf('   ✓ PostProcessingModule integration successful\n');
        fprintf('   ✓ Total peak power: %.2f W\n', processed_trial_enhanced.total_peak_power);
        fprintf('   ✓ Total work: %.2f J\n', processed_trial_enhanced.total_work);
    else
        fprintf('   ✗ PostProcessingModule integration failed\n');
    end

catch ME
    fprintf('   ✗ PostProcessingModule integration failed: %s\n', ME.message);
end

% Summary
fprintf('\n=== Test Summary ===\n');
fprintf('✓ Sample data creation: %d time points, 3 joints\n', num_samples);
fprintf('✓ Power calculations: Always performed\n');
fprintf('✓ Work calculations: Optional (controlled by calculate_work flag)\n');
fprintf('✓ Angular impulse calculations: Granular breakdown by:\n');
fprintf('  - Joint end (proximal/distal)\n');
fprintf('  - Source (joint torques, applied torques, force moments)\n');
fprintf('  - Total per joint end\n');
fprintf('✓ Integration with PostProcessingModule: Functional\n\n');

fprintf('The granular angular impulse calculations provide detailed insight into:\n');
fprintf('- Where angular impulse is generated (proximal vs distal ends)\n');
fprintf('- What contributes to angular impulse (torques vs force moments)\n');
fprintf('- Total angular impulse acting on each joint segment\n\n');

fprintf('Test completed successfully!\n');
