function [joint_moments, segment_data] = calculateForceMoments(simulation_data)
%CALCULATEFORCEMOMENTS Calculate moments of forces at joints on segments
%
% This function calculates the moments created by joint forces about
% segment centers of mass and other reference points

% Initialize output structures
joint_moments = struct();
segment_data = struct();

% Define segments and their associated joints
segments = {
    'LeftUpperArm', {'LS', 'LE'};     % Left shoulder and elbow
    'RightUpperArm', {'RS', 'RE'};   % Right shoulder and elbow
    'LeftForearm', {'LE', 'LW'};     % Left elbow and wrist
    'RightForearm', {'RE', 'RW'};    % Right elbow and wrist
    'UpperTorso', {'Spine', 'LS', 'RS'}; % Spine and shoulders
    'LowerTorso', {'Hip', 'Spine'};  % Hip and spine
};

fprintf('Calculating force moments on segments...\n');

for i = 1:size(segments, 1)
    segment_name = segments{i, 1};
    joint_list = segments{i, 2};

    fprintf('Processing %s segment...\n', segment_name);

    % Calculate moments for this segment
    moments = calculateSegmentMoments(simulation_data, segment_name, joint_list);

    % Store results
    joint_moments.(segment_name) = moments;

    % Add to export data
    segment_data = addSegmentMomentsToExport(segment_data, segment_name, moments);
end

fprintf('Force moment calculations complete.\n');

end

function moments = calculateSegmentMoments(sim_data, segment_name, joint_list)
%CALCULATESEGMENTMOMENTS Calculate moments for a specific segment

moments = struct();

% Get segment properties
segment_props = getSegmentProperties(sim_data, segment_name);

for j = 1:length(joint_list)
    joint_name = joint_list{j};

    % Get joint forces (global coordinates)
    forces_global = getJointForcesGlobal(sim_data, joint_name);

    % Get joint position relative to segment COM
    joint_pos = getJointPosition(sim_data, joint_name);
    segment_com = segment_props.center_of_mass;

    % Calculate position vector from COM to joint
    r_vector = joint_pos - segment_com;

    % Calculate moment: M = r × F
    moment = calculateCrossProduct(r_vector, forces_global);

    % Store moment
    moments.([joint_name '_moment_about_COM']) = moment;

    % Also calculate moment about joint itself (for segment analysis)
    if j > 1
        prev_joint = joint_list{j-1};
        prev_joint_pos = getJointPosition(sim_data, prev_joint);
        r_joint = joint_pos - prev_joint_pos;
        moment_joint = calculateCrossProduct(r_joint, forces_global);
        moments.([joint_name '_moment_about_' prev_joint]) = moment_joint;
    end
end

% Calculate net moment on segment
moments.net_moment = calculateNetMoment(moments);

end

function forces_global = getJointForcesGlobal(sim_data, joint_name)
%GETJOINTFORCESGLOBAL Extract joint forces in global coordinates

% Check if global forces are available
global_force_fields = {
    [joint_name 'Logs_ForceGlobal_1'];
    [joint_name 'Logs_ForceGlobal_2'];
    [joint_name 'Logs_ForceGlobal_3']
};

% If global forces exist, use them
if isfield(sim_data, global_force_fields{1})
    forces_global = [sim_data.(global_force_fields{1}), ...
                     sim_data.(global_force_fields{2}), ...
                     sim_data.(global_force_fields{3})];
else
    % Convert local forces to global
    forces_local = getJointForcesLocal(sim_data, joint_name);
    R_matrices = getRotationMatrices(sim_data, joint_name);

    n_samples = size(forces_local, 1);
    forces_global = zeros(size(forces_local));

    for i = 1:n_samples
        R = squeeze(R_matrices(i, :, :));
        forces_global(i, :) = (R * forces_local(i, :)')';
    end
end

end

function forces_local = getJointForcesLocal(sim_data, joint_name)
%GETJOINTFORCESLOCAL Extract local joint forces

% Try different field naming conventions
field_patterns = {
    [joint_name 'Logs_ForceLocal_'];
    [joint_name 'Logs_ConstraintForceLocal_'];
    [joint_name 'Logs_TotalForceLocal_']
};

for p = 1:length(field_patterns)
    field_base = field_patterns{p};
    if isfield(sim_data, [field_base '1'])
        forces_local = [sim_data.([field_base '1']), ...
                        sim_data.([field_base '2']), ...
                        sim_data.([field_base '3'])];
        return;
    end
end

% If no force fields found, return zeros
warning('No force fields found for joint %s', joint_name);
forces_local = zeros(length(sim_data.time), 3);

end

function joint_pos = getJointPosition(sim_data, joint_name)
%GETJOINTPOSITION Get joint position in global coordinates

field_base = [joint_name 'Logs_GlobalPosition_'];

if isfield(sim_data, [field_base '1'])
    joint_pos = [sim_data.([field_base '1']), ...
                 sim_data.([field_base '2']), ...
                 sim_data.([field_base '3'])];
else
    % Alternative field names
    alt_fields = {
        [joint_name 'Logs_Position_'];
        [joint_name '_GlobalPosition_']
    };

    for a = 1:length(alt_fields)
        if isfield(sim_data, [alt_fields{a} '1'])
            joint_pos = [sim_data.([alt_fields{a} '1']), ...
                         sim_data.([alt_fields{a} '2']), ...
                         sim_data.([alt_fields{a} '3'])];
            return;
        end
    end

    warning('Position not found for joint %s', joint_name);
    joint_pos = zeros(length(sim_data.time), 3);
end

end

function segment_props = getSegmentProperties(sim_data, segment_name)
%GETSEGMENTPROPERTIES Extract segment properties (COM, inertia, etc.)

segment_props = struct();

% Map segment names to data fields
switch segment_name
    case 'LeftUpperArm'
        if isfield(sim_data, 'SegmentInertiaLogs_LeftUpperArmCOM_dim1')
            segment_props.center_of_mass = [
                sim_data.SegmentInertiaLogs_LeftUpperArmCOM_dim1, ...
                sim_data.SegmentInertiaLogs_LeftUpperArmCOM_dim2, ...
                sim_data.SegmentInertiaLogs_LeftUpperArmCOM_dim3
            ];
        end

    case 'RightUpperArm'
        if isfield(sim_data, 'SegmentInertiaLogs_RightUpperArmCOM_dim1')
            segment_props.center_of_mass = [
                sim_data.SegmentInertiaLogs_RightUpperArmCOM_dim1, ...
                sim_data.SegmentInertiaLogs_RightUpperArmCOM_dim2, ...
                sim_data.SegmentInertiaLogs_RightUpperArmCOM_dim3
            ];
        end

    case 'LeftForearm'
        if isfield(sim_data, 'SegmentInertiaLogs_LFLowerCOM_dim1')
            segment_props.center_of_mass = [
                sim_data.SegmentInertiaLogs_LFLowerCOM_dim1, ...
                sim_data.SegmentInertiaLogs_LFLowerCOM_dim2, ...
                sim_data.SegmentInertiaLogs_LFLowerCOM_dim3
            ];
        end

    case 'RightForearm'
        if isfield(sim_data, 'SegmentInertiaLogs_RFLowerCOM_dim1')
            segment_props.center_of_mass = [
                sim_data.SegmentInertiaLogs_RFLowerCOM_dim1, ...
                sim_data.SegmentInertiaLogs_RFLowerCOM_dim2, ...
                sim_data.SegmentInertiaLogs_RFLowerCOM_dim3
            ];
        end

    case 'UpperTorso'
        if isfield(sim_data, 'SegmentInertiaLogs_UpperTorsoCOM_dim1')
            segment_props.center_of_mass = [
                sim_data.SegmentInertiaLogs_UpperTorsoCOM_dim1, ...
                sim_data.SegmentInertiaLogs_UpperTorsoCOM_dim2, ...
                sim_data.SegmentInertiaLogs_UpperTorsoCOM_dim3
            ];
        end

    case 'LowerTorso'
        if isfield(sim_data, 'SegmentInertiaLogs_LowerTorsoCOM_dim1')
            segment_props.center_of_mass = [
                sim_data.SegmentInertiaLogs_LowerTorsoCOM_dim1, ...
                sim_data.SegmentInertiaLogs_LowerTorsoCOM_dim2, ...
                sim_data.SegmentInertiaLogs_LowerTorsoCOM_dim3
            ];
        end

    otherwise
        warning('Unknown segment: %s', segment_name);
        segment_props.center_of_mass = zeros(length(sim_data.time), 3);
end

% If COM not found, use zeros
if ~isfield(segment_props, 'center_of_mass')
    segment_props.center_of_mass = zeros(length(sim_data.time), 3);
end

end

function moment = calculateCrossProduct(r_vector, force_vector)
%CALCULATECROSSPRODUCT Calculate cross product M = r × F

n_samples = size(r_vector, 1);
moment = zeros(n_samples, 3);

for i = 1:n_samples
    r = r_vector(i, :);
    f = force_vector(i, :);

    % Cross product: r × f
    moment(i, :) = [
        r(2)*f(3) - r(3)*f(2);  % Mx
        r(3)*f(1) - r(1)*f(3);  % My
        r(1)*f(2) - r(2)*f(1)   % Mz
    ];
end

end

function net_moment = calculateNetMoment(moments)
%CALCULATENETMOMENT Calculate net moment on segment

moment_fields = fieldnames(moments);
net_moment = zeros(size(moments.(moment_fields{1})));

for i = 1:length(moment_fields)
    if contains(moment_fields{i}, 'moment_about_COM')
        net_moment = net_moment + moments.(moment_fields{i});
    end
end

end

function segment_data = addSegmentMomentsToExport(segment_data, segment_name, moments)
%ADDSEGMENTMOMENTSTOEXPORT Add segment moment data to export structure

moment_fields = fieldnames(moments);

for i = 1:length(moment_fields)
    field_name = [segment_name '_' moment_fields{i}];
    moment_data = moments.(moment_fields{i});

    if size(moment_data, 2) == 3
        % 3D moment vector
        segment_data.([field_name '_X']) = moment_data(:, 1);
        segment_data.([field_name '_Y']) = moment_data(:, 2);
        segment_data.([field_name '_Z']) = moment_data(:, 3);

        % Magnitude
        segment_data.([field_name '_Magnitude']) = sqrt(sum(moment_data.^2, 2));
    else
        % Scalar moment
        segment_data.(field_name) = moment_data;
    end
end

end

function R_matrices = getRotationMatrices(sim_data, joint_name)
%GETROTATIONMATRICES Extract rotation matrices for coordinate transformation
% (This function is duplicated from the power calculation file for completeness)

field_base = [joint_name 'Logs_Rotation_Transform_'];

% Check if rotation matrix fields exist
if ~isfield(sim_data, [field_base 'I11'])
    warning('Rotation matrices not found for %s', joint_name);
    n_samples = length(sim_data.time);
    R_matrices = repmat(eye(3), [1, 1, n_samples]);
    R_matrices = permute(R_matrices, [3, 1, 2]);
    return;
end

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

%% Example usage and integration with main power calculation:
%
% % Main calculation function
% function enhanced_data = processGolfSwingData(simulation_data)
%
%     % Calculate joint power and work
%     [joint_powers, joint_work, joint_data] = calculateJointPowerWork(simulation_data);
%
%     % Calculate force moments
%     [joint_moments, segment_data] = calculateForceMoments(simulation_data);
%
%     % Combine all data
%     enhanced_data = simulation_data;
%
%     % Add joint power/work data
%     joint_fields = fieldnames(joint_data);
%     for i = 1:length(joint_fields)
%         enhanced_data.(joint_fields{i}) = joint_data.(joint_fields{i});
%     end
%
%     % Add segment moment data
%     segment_fields = fieldnames(segment_data);
%     for i = 1:length(segment_fields)
%         enhanced_data.(segment_fields{i}) = segment_data.(segment_fields{i});
%     end
%
%     fprintf('Enhanced dataset contains %d variables\n', length(fieldnames(enhanced_data)));
%
% end
