function export_two_hand_wscg_tables(output_directory)
%EXPORT_TWO_HAND_WSCG_TABLES Export a portable two-hand evidence cache.
%
% This function reads the archived WSCG BASE, ZTCF, and DELTA MATLAB tables
% and writes only the columns required for the planar wrench audit.  The source
% MAT files remain unchanged.  Run from the repository root with, for example:
%
%   export_two_hand_wscg_tables('docs/research/.../data/wscg_two_hand_raw')

arguments
    output_directory (1, 1) string
end

repository_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
table_directory = fullfile(repository_root, 'src', 'engines', ...
    'Simscape_Multibody_Models', '2D_Golf_Model', 'matlab', ...
    'Model Output', 'Tables');

if ~isfolder(output_directory)
    mkdir(output_directory);
end

case_names = ["BASE", "ZTCF", "DELTA"];
for case_name = case_names
    source_path = fullfile(table_directory, case_name + ".mat");
    loaded = load(source_path);
    source = loaded.(case_name);
    portable = select_columns(source);
    output_path = fullfile(output_directory, lower(case_name) + ".csv");
    writetable(portable, output_path);
end
end


function output = select_columns(source)
%SELECT_COLUMNS Flatten vector-valued table variables with declared SI labels.
output = table(source.Time, 'VariableNames', {'time_s'});

output = add_vector(output, 'lead_position', source.LWGlobalPosition, 'm');
output = add_vector(output, 'trail_position', source.RWGlobalPosition, 'm');
output = add_vector(output, 'midpoint_position', ...
    [source.MPx, source.MPy, source.MPz], 'm');
output = add_vector(output, 'lead_force_global', source.LWonClubFGlobal, 'n');
output = add_vector(output, 'trail_force_global', source.RWonClubFGlobal, 'n');
output = add_vector(output, 'lead_force_local_source', ...
    source.LeftWristForceLocal, 'n');
output = add_vector(output, 'trail_force_local_source', ...
    source.RightWristForceLocal, 'n');
output = add_vector(output, 'lead_free_torque_global', ...
    source.LWonClubTGlobal, 'nm');
output = add_vector(output, 'trail_free_torque_global', ...
    source.RWonClubTGlobal, 'nm');
output = add_vector(output, 'lead_velocity_global', ...
    source.LeftHandVelocity, 'm_s');
output = add_vector(output, 'trail_velocity_global', ...
    source.RightHandVelocity, 'm_s');
output = add_vector(output, 'resultant_force_global_source', ...
    source.TotalHandForceGlobal, 'n');
output = add_vector(output, 'resultant_force_local_source', ...
    source.TotalHandForceonClubLocal, 'n');
output = add_vector(output, 'equivalent_midpoint_couple_global_source', ...
    source.EquivalentMidpointCoupleGlobal, 'nm');
output = add_vector(output, 'equivalent_midpoint_couple_local_source', ...
    source.EquivalentMidpointCoupleLocal, 'nm');
output = add_vector(output, 'lead_force_moment_midpoint_local_source', ...
    source.LHMOFonClubLocal, 'nm');
output = add_vector(output, 'trail_force_moment_midpoint_local_source', ...
    source.RHMOFonClubLocal, 'nm');
output = add_vector(output, 'force_moment_midpoint_local_source', ...
    source.MPMOFonClubLocal, 'nm');
output = add_vector(output, 'total_free_torque_global_source', ...
    source.TotalWristTorqueGlobal, 'nm');
output = add_vector(output, 'total_free_torque_local_source', ...
    source.TotalWristTorqueLocal, 'nm');

output = add_vector(output, 'midpoint_angular_velocity_global', ...
    source.MidpointGlobalAV, 'deg_s');
output.club_handle_angular_velocity_deg_s = source.ClubhandleAV;
output.clubhead_speed_mph = source.("CHS (mph)");
output.angle_of_attack_deg = source.AoA;
output.lag_angle_rad = source.LagAngleRadians;
output.lead_linear_power_w = source.LWonClubLinearPower;
output.trail_linear_power_w = source.RWonClubLinearPower;
output.lead_angular_power_w = source.LWonClubAngularPower;
output.trail_angular_power_w = source.RWonClubAngularPower;
output.killswitch_state = source.KillswitchState;
output.kill_damping_state = source.KillDampState;
output.lead_command_torque_nm = source.JointTorqueLWrist;
output.trail_command_torque_nm = source.JointTorqueRWrist;

assert(height(output) == height(source), 'Export changed the number of rows.');
numeric_data = output{:,:};
assert(all(isfinite(numeric_data), 'all'), ...
    'Portable evidence cache contains non-finite values.');
end


function output = add_vector(output, stem, values, unit)
%ADD_VECTOR Append x, y, and z columns from one N-by-3 vector variable.
assert(size(values, 2) == 3, stem + " must be an N-by-3 variable.");
axes = ["x", "y", "z"];
for index = 1:3
    name = stem + "_" + axes(index) + "_" + unit;
    output.(name) = values(:, index);
end
end
