function [joint_powers, joint_work, joint_data] = calculateJointPowerWork(simulation_data)
%CALCULATEJOINTPOWERWORK Calculate power and work at all joints
%
% Inputs:
%   simulation_data - Structure containing all simulation outputs
%
% Outputs:
%   joint_powers - Structure with power time series for each joint
%   joint_work - Structure with cumulative work for each joint
%   joint_data - Structure with processed joint data for export

% Time vector
time = simulation_data.time;
dt = mean(diff(time)); % Time step

% Initialize output structures
joint_powers = struct();
joint_work = struct();
joint_data = struct();

% Define joint list and their DOF
joints = {
    'LScap', 2;  % Left Scapula (X,Y)
    'RScap', 2;  % Right Scapula (X,Y)
    'LS', 3;     % Left Shoulder (X,Y,Z)
    'RS', 3;     % Right Shoulder (X,Y,Z)
    'LF', 1;     % Left Forearm (Z)
    'RF', 1;     % Right Forearm (Z)
    'Spine', 2;  % Spine (X,Y)
    'Torso', 1;  % Torso (Z)
};

fprintf('Calculating joint power and work...\n');

for i = 1:size(joints, 1)
    joint_name = joints{i, 1};
    dof = joints{i, 2};

    fprintf('Processing %s joint (%d DOF)...\n', joint_name, dof);

    % Extract data for this joint
    [power_data, work_data] = processJoint(simulation_data, joint_name, dof, dt);

    % Store results
    joint_powers.(joint_name) = power_data;
    joint_work.(joint_name) = work_data;

    % Prepare data for export (flatten for CSV)
    joint_data = addJointToExport(joint_data, joint_name, power_data, work_data);
end

fprintf('Joint power and work calculations complete.\n');

end

function [power_data, work_data] = processJoint(sim_data, joint_name, dof, dt)
%PROCESSJOINT Calculate power and work for a specific joint

% Initialize structures
power_data = struct();
work_data = struct();

% Get angular velocities (global coordinates)
omega = getGlobalAngularVelocities(sim_data, joint_name, dof);

% Get actuator torques
tau_actuator = getActuatorTorques(sim_data, joint_name, dof);

% Get constraint torques (convert to global if needed)
tau_constraint_global = getConstraintTorquesGlobal(sim_data, joint_name, dof);

% Calculate powers
power_data.actuator = calculatePower(tau_actuator, omega);
power_data.constraint = calculatePower(tau_constraint_global, omega);
power_data.total = power_data.actuator + power_data.constraint;

% Calculate work (cumulative integration)
work_data.actuator = cumtrapz(power_data.actuator) * dt;
work_data.constraint = cumtrapz(power_data.constraint) * dt;
work_data.total = cumtrapz(power_data.total) * dt;

% Additional metrics
work_data.positive_work = cumtrapz(max(power_data.total, 0)) * dt;
work_data.negative_work = cumtrapz(min(power_data.total, 0)) * dt;

end

function omega = getGlobalAngularVelocities(sim_data, joint_name, dof)
%GETGLOBALANGULARVELOCITIES Extract global angular velocities

switch joint_name
    case {'LScap', 'RScap'}
        % 2-DOF joints
        field_base = [joint_name 'Logs_GlobalAngularVelocity_'];
        omega = [sim_data.([field_base '1']), ...
                 sim_data.([field_base '2'])];
        if dof == 2
            omega = omega(:, 1:2);
        end

    case {'LS', 'RS'}
        % 3-DOF joints
        field_base = [joint_name 'Logs_GlobalAngularVelocity_'];
        omega = [sim_data.([field_base '1']), ...
                 sim_data.([field_base '2']), ...
                 sim_data.([field_base '3'])];

    case {'LF', 'RF'}
        % 1-DOF joints (only Z component typically)
        field_base = [joint_name 'Logs_GlobalAngularVelocity_'];
        omega = sim_data.([field_base '3']); % Z-component

    case 'Spine'
        % 2-DOF joint
        omega = [sim_data.SpineLogs_GlobalAngularVelocity_1, ...
                 sim_data.SpineLogs_GlobalAngularVelocity_2];

    case 'Torso'
        % 1-DOF joint
        omega = sim_data.TorsoLogs_GlobalAngularVelocity_3; % Z-rotation

    otherwise
        error('Unknown joint: %s', joint_name);
end

end

function tau = getActuatorTorques(sim_data, joint_name, dof)
%GETACTUATORTORQUES Extract actuator torques
% Note: These may need coordinate transformation if not in global coordinates

switch joint_name
    case {'LScap', 'RScap'}
        % Assuming actuator torques are in global coordinates
        tau = [sim_data.([joint_name 'Logs_ActuatorTorqueX']), ...
               sim_data.([joint_name 'Logs_ActuatorTorqueY'])];

    case {'LS', 'RS'}
        tau = [sim_data.([joint_name 'Logs_ActuatorTorqueX']), ...
               sim_data.([joint_name 'Logs_ActuatorTorqueY']), ...
               sim_data.([joint_name 'Logs_ActuatorTorqueZ'])];

    case {'LF', 'RF'}
        % Single DOF - usually Z torque
        tau = sim_data.([joint_name 'Logs_ActuatorTorqueZ']);

    case 'Spine'
        tau = [sim_data.SpineLogs_ActuatorTorqueX, ...
               sim_data.SpineLogs_ActuatorTorqueY];

    case 'Torso'
        % Check if torso has actuator torque field
        if isfield(sim_data, 'TorsoLogs_ActuatorTorque')
            tau = sim_data.TorsoLogs_ActuatorTorque;
        else
            tau = zeros(size(sim_data.time)); % No actuator torque
        end

    otherwise
        error('Unknown joint: %s', joint_name);
end

end

function tau_global = getConstraintTorquesGlobal(sim_data, joint_name, dof)
%GETCONSTRAINTTORQUESGLOBAL Convert constraint torques to global coordinates

% Get local constraint torques
tau_local = getConstraintTorquesLocal(sim_data, joint_name, dof);

% Get rotation matrices for coordinate transformation
R_matrices = getRotationMatrices(sim_data, joint_name);

% Transform to global coordinates
n_samples = size(tau_local, 1);
tau_global = zeros(size(tau_local));

for i = 1:n_samples
    if dof == 1
        % For 1-DOF joints, may need special handling
        tau_global(i) = tau_local(i); % Assuming already aligned
    else
        % Transform: τ_global = R * τ_local
        R = squeeze(R_matrices(i, :, :));
        tau_global(i, :) = (R * tau_local(i, :)')';
    end
end

end

function tau_local = getConstraintTorquesLocal(sim_data, joint_name, dof)
%GETCONSTRAINTTORQUESLOCAL Extract local constraint torques

field_base = [joint_name 'Logs_ConstraintTorqueLocal_'];

switch dof
    case 1
        tau_local = sim_data.([field_base '3']); % Usually Z for 1-DOF
    case 2
        tau_local = [sim_data.([field_base '1']), ...
                     sim_data.([field_base '2'])];
    case 3
        tau_local = [sim_data.([field_base '1']), ...
                     sim_data.([field_base '2']), ...
                     sim_data.([field_base '3'])];
end

end

function R_matrices = getRotationMatrices(sim_data, joint_name)
%GETROTATIONMATRICES Extract rotation matrices for coordinate transformation

field_base = [joint_name 'Logs_Rotation_Transform_'];

% Extract 3x3 rotation matrix elements
I11 = sim_data.([field_base 'I11']);
I12 = sim_data.([field_base 'I12']);
I13 = sim_data.([field_base 'I13']);
I21 = sim_data.([field_base 'I21']);
I22 = sim_data.([field_base 'I22']);
I23 = sim_data.([field_base 'I23']);
I31 = sim_data.([field_base 'I31']);
I32 = sim_data.([field_base 'I32']);
I33 = sim_data.([field_base 'I33']);

n_samples = length(I11);
R_matrices = zeros(n_samples, 3, 3);

for i = 1:n_samples
    R_matrices(i, :, :) = [I11(i), I12(i), I13(i);
                           I21(i), I22(i), I23(i);
                           I31(i), I32(i), I33(i)];
end

end

function power = calculatePower(torque, angular_velocity)
%CALCULATEPOWER Calculate power from torque and angular velocity

if size(torque, 2) == 1 && size(angular_velocity, 2) == 1
    % 1-DOF case
    power = torque .* angular_velocity;
else
    % Multi-DOF case: P = τ · ω (dot product)
    power = sum(torque .* angular_velocity, 2);
end

end

function joint_data = addJointToExport(joint_data, joint_name, power_data, work_data)
%ADDJOINTTOEXPORT Add joint data to export structure

% Add power data
joint_data.([joint_name '_Power_Actuator']) = power_data.actuator;
joint_data.([joint_name '_Power_Constraint']) = power_data.constraint;
joint_data.([joint_name '_Power_Total']) = power_data.total;

% Add work data
joint_data.([joint_name '_Work_Actuator']) = work_data.actuator;
joint_data.([joint_name '_Work_Constraint']) = work_data.constraint;
joint_data.([joint_name '_Work_Total']) = work_data.total;
joint_data.([joint_name '_Work_Positive']) = work_data.positive_work;
joint_data.([joint_name '_Work_Negative']) = work_data.negative_work;

end

%% Example usage:
% [powers, work, export_data] = calculateJointPowerWork(simulation_data);
%
% % Add to existing data for export
% enhanced_data = [simulation_data, export_data];
%
% % Export to CSV
% writetable(struct2table(enhanced_data), 'enhanced_golf_data.csv');
