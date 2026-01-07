
% Postprocessing Script for Auxiliary Golf Swing Metrics (Standard Workspace Variables)
% Assumes time, positions, forces, and velocities have been logged from Simulink

% === Load Required Signals ===
% Replace these with your actual logged variable names
t = time_vector; % Time vector [Nx1]
CH_pos = [CHx, CHy, CHz];              % Clubhead position [Nx3]
CH_vel = [CHvx_x, CHvx_y, CHvx_z];     % Clubhead velocity [Nx3]
LH_force = LH_force_global;            % Left hand force [Nx3]
RH_force = RH_force_global;            % Right hand force [Nx3]

% === Clubhead Speed (CHS) ===
CHS = vecnorm(CH_vel, 2, 2);       % m/s
CHS_mph = CHS * 2.23694;

% === Clubhead Path Unit Vector ===
CH_path_vec = diff(CH_pos) ./ diff(t);
CH_path_unit = CH_path_vec ./ vecnorm(CH_path_vec, 2, 2);

% === Angle of Attack (AoA) ===
AoA_rad = atan2(CH_vel(:,3), CH_vel(:,1));
AoA_deg = rad2deg(AoA_rad);

% === Work and Power Calculations ===
LH_power = dot(LH_force, CH_vel, 2);
RH_power = dot(RH_force, CH_vel, 2);
LinearPoweronClub = LH_power + RH_power;

% Cumulative work (trapezoidal integration)
LH_work = cumtrapz(t, LH_power);
RH_work = cumtrapz(t, RH_power);
LinearWorkonClub = LH_work + RH_work;

% === Impulse (linear) ===
LH_impulse = cumtrapz(t, LH_force);
RH_impulse = cumtrapz(t, RH_force);

% === Save or Export Results ===
save('golf_post_results.mat', 'CHS_mph', 'AoA_deg', 'LinearPoweronClub', ...
     'LH_work', 'RH_work', 'LH_impulse', 'RH_impulse', 'CH_path_unit');
