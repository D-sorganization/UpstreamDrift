
% Postprocessing Script using Simscape Logging (simlog)
% Assumes 'simlog_GolfSwing3D_KineticallyDriven' exists in the workspace

% === Time Vector ===
t = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHx.series.time;

% === Clubhead Position and Velocity ===
CHx = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHx.series.values;
CHy = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHy.series.values;
CHz = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHz.series.values;
CH_pos = [CHx, CHy, CHz];

CHvx = simlog_GolfSwing3D_KineticallyDriven.Clubhead.CHvx.series.values;
CH_vel = CHvx; % [Nx3]

% === Clubhead Speed (CHS) ===
CHS = vecnorm(CH_vel, 2, 2);       % m/s
CHS_mph = CHS * 2.23694;

% === Clubhead Path Unit Vector ===
CH_path_vec = diff(CH_pos) ./ diff(t);
CH_path_unit = CH_path_vec ./ vecnorm(CH_path_vec, 2, 2);

% === Angle of Attack (AoA) ===
AoA_rad = atan2(CH_vel(:,3), CH_vel(:,1));
AoA_deg = rad2deg(AoA_rad);

% === Forces from Hands ===
LH_force = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter.series.values;
RH_force = simlog_GolfSwing3D_KineticallyDriven.Grip.PS_Simulink_Converter2.series.values;

% === Work and Power Calculations ===
LH_power = dot(LH_force, CH_vel, 2);
RH_power = dot(RH_force, CH_vel, 2);
LinearPoweronClub = LH_power + RH_power;

% Cumulative work
LH_work = cumtrapz(t, LH_power);
RH_work = cumtrapz(t, RH_power);
LinearWorkonClub = LH_work + RH_work;

% === Impulse (linear) ===
LH_impulse = cumtrapz(t, LH_force);
RH_impulse = cumtrapz(t, RH_force);

% === Save or Export Results ===
save('golf_post_simlog_results.mat', 'CHS_mph', 'AoA_deg', 'LinearPoweronClub', ...
     'LH_work', 'RH_work', 'LH_impulse', 'RH_impulse', 'CH_path_unit');
