
% Postprocessing Script using Simscape Logging with Rotation Matrix-derived Angular Velocity
% Assumes 'simlog_GolfSwing3D_KineticallyDriven' exists in the workspace

% === Time Vector ===
t = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHx.series.time;
dt = [diff(t); diff(t(end-1:end))]; % ensure same length

% === Clubhead Position and Velocity ===
CHx = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHx.series.values;
CHy = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHy.series.values;
CHz = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHz.series.values;
CH_pos = [CHx, CHy, CHz];

CHvx = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHvx.series.values;
CH_vel = CHvx;

% === Clubhead Speed and AoA ===
CHS = vecnorm(CH_vel, 2, 2);
CHS_mph = CHS * 2.23694;
AoA_rad = atan2(CH_vel(:,3), CH_vel(:,1));
AoA_deg = rad2deg(AoA_rad);

% === Clubhead Path Unit Vector ===
CH_path_vec = diff(CH_pos) ./ diff(t);
CH_path_unit = CH_path_vec ./ vecnorm(CH_path_vec, 2, 2);

% === Angular Velocity from Rotation Matrices ===
% Example assumes rotation matrix R stored as a flattened 3x3 matrix [R11, R12, ..., R33] over time
% This should be replaced with your actual logged rotation matrix data
% For illustration, we simulate Rdata with identity matrices
n = length(t);
Rdata = repmat(reshape(eye(3), 1, 9), n, 1);  % Placeholder for R matrix log
omega = zeros(n, 3);  % Initialize angular velocity

for k = 2:n
    R_prev = reshape(Rdata(k-1, :), [3, 3]);
    R_curr = reshape(Rdata(k, :), [3, 3]);
    R_dot = (R_curr - R_prev) / dt(k);
    omega_mat = R_curr' * R_dot;
    omega(k, :) = [omega_mat(3,2); omega_mat(1,3); omega_mat(2,1)]; % skew-symmetric extract
end

% === Hand Forces and Torques ===
LH_force = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter.series.values;
RH_force = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter2.series.values;
LH_torque = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter1.series.values;
RH_torque = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter3.series.values;

% === Power ===
LH_power = dot(LH_force, CH_vel, 2);
RH_power = dot(RH_force, CH_vel, 2);
LinearPoweronClub = LH_power + RH_power;

LH_ang_power = dot(LH_torque, omega, 2);
RH_ang_power = dot(RH_torque, omega, 2);

% === Work and Impulse ===
LH_work = cumtrapz(t, LH_power);
RH_work = cumtrapz(t, RH_power);
LH_ang_work = cumtrapz(t, LH_ang_power);
RH_ang_work = cumtrapz(t, RH_ang_power);

LH_impulse = cumtrapz(t, LH_force);
RH_impulse = cumtrapz(t, RH_force);

% === Save Results ===
save('golf_post_simlog_rotmat_results.mat', 'CHS_mph', 'AoA_deg', 'CH_path_unit', ...
     'LH_power', 'RH_power', 'LinearPoweronClub', ...
     'LH_ang_power', 'RH_ang_power', ...
     'LH_work', 'RH_work', 'LH_ang_work', 'RH_ang_work', ...
     'LH_impulse', 'RH_impulse', 'omega');
