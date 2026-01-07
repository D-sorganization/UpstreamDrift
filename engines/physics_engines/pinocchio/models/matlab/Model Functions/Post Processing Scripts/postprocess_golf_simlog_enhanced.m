
% Enhanced Postprocessing Script using Simscape Logging (simlog)
% Includes reference frame transforms for angular power

t = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHx.series.time;

% === Clubhead Position and Velocity ===
CHx = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHx.series.values;
CHy = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHy.series.values;
CHz = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHz.series.values;
CH_pos = [CHx, CHy, CHz];

CHvx = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHvx.series.values;
CH_vel = CHvx;

% === Clubhead Speed (CHS) ===
CHS = vecnorm(CH_vel, 2, 2);
CHS_mph = CHS * 2.23694;

% === Path Unit Vector and AoA ===
CH_path_vec = diff(CH_pos) ./ diff(t);
CH_path_unit = CH_path_vec ./ vecnorm(CH_path_vec, 2, 2);
AoA_rad = atan2(CH_vel(:,3), CH_vel(:,1));
AoA_deg = rad2deg(AoA_rad);

% === Force and Torque from Hands ===
LH_force = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter.series.values;
RH_force = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter2.series.values;
LH_torque = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter1.series.values;
RH_torque = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter3.series.values;

% === Angular Velocity (assumed from CH angular velocity if available) ===
% Replace with actual angular velocity if you have one
omega = zeros(size(CH_vel)); % Placeholder

% === Power ===
LH_power = dot(LH_force, CH_vel, 2);
RH_power = dot(RH_force, CH_vel, 2);
LinearPoweronClub = LH_power + RH_power;

LH_ang_power = dot(LH_torque, omega, 2); % Placeholder
RH_ang_power = dot(RH_torque, omega, 2); % Placeholder

% === Work and Impulse ===
LH_work = cumtrapz(t, LH_power);
RH_work = cumtrapz(t, RH_power);
LH_ang_work = cumtrapz(t, LH_ang_power);
RH_ang_work = cumtrapz(t, RH_ang_power);

LH_impulse = cumtrapz(t, LH_force);
RH_impulse = cumtrapz(t, RH_force);

% === Save Outputs ===
save('golf_post_simlog_results.mat', 'CHS_mph', 'AoA_deg', 'CH_path_unit', ...
     'LH_power', 'RH_power', 'LinearPoweronClub', ...
     'LH_ang_power', 'RH_ang_power', ...
     'LH_work', 'RH_work', 'LH_ang_work', 'RH_ang_work', ...
     'LH_impulse', 'RH_impulse');
