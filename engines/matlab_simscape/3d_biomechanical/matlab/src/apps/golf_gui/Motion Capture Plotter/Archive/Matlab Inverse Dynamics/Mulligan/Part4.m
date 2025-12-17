%% === Helper: Update Offset ===
function updateOffset(src)
    handles = guidata(src);
    val = get(src, 'Value');
    opts = get(src, 'String');
    inch_offset = str2double(opts{val});
    handles.evalOffset = inch_offset * 0.0254;  % Convert inches to meters
    guidata(src, handles);
    updateFrame(handles.frame);
end

%% === Helper: Select Filter ===
function selectFilter(src, selected)
    handles = guidata(src);
    for k = 1:length(handles.filterChecks)
        if handles.filterChecks(k) ~= src
            set(handles.filterChecks(k), 'Value', 0);
        end
    end
    handles.currentFilter = selected;
    guidata(src, handles);
    updateFrame(handles.frame);
end

%% === Apply Filter ===
function filtered = applyFilter(data, method)
    switch method
        case 'None'
            filtered = data;
        case 'MovingAvg'
            filtered = movmean(data, 5);
        case 'SavitzkyGolay'
            filtered = sgolayfilt(data, 3, 9);
        case {'Butter6','Butter8','Butter10','Butter12'}
            cutoff = str2double(extractAfter(method,'Butter'));
            [b,a] = butter(4, cutoff/30, 'low');
            filtered = filtfilt(b, a, data);
        case 'QuinticSpline'
            t = linspace(0,1,size(data,1));
            filtered = zeros(size(data));
            for j = 1:size(data,2)
                [pp,~] = spaps(t, data(:,j)', 1e-6);
                filtered(:,j) = fnval(pp, t)';
            end
        case 'Lowess'
            filtered = zeros(size(data));
            for j = 1:size(data,2)
                filtered(:,j) = smooth(data(:,j), 0.1, 'lowess');
            end
        otherwise
            error('Unknown filter: %s', method);
    end
end

%% === Compute Kinematics ===
function [acc, alpha, omega] = computeKinematics(pos, ori, t)
    dt = mean(diff(t));
    vel = gradient(pos, dt);
    acc = gradient(vel, dt);
    omega = gradient(ori, dt);
    alpha = gradient(omega, dt);
end

%% === Compute Dynamics ===
function [F, Tau, evalPt, shaftDir, cgPt] = computeDynamics(pos, ori, frame, offset, I, m)
    % Extract shaft direction and define eval point and CG
    midhands = pos(frame,1:3);
    clubface = pos(frame,4:6);
    shaftVec = clubface - midhands;
    shaftDir = shaftVec / norm(shaftVec);

    % Define evaluation point (where F is applied)
    evalPt = midhands + shaftDir * offset;

    % Define CG (10 cm behind clubface)
    cgPt = clubface - shaftDir * 0.10;

    % Filtered orientation and position
    [accel, alpha, omega] = computeKinematics(pos(:,1:3), ori, linspace(0,1,size(pos,1)));

    % Force = mass * linear acceleration
    F = m * accel(frame, :);

    % Moment arm from eval point to CG
    r = cgPt - evalPt;

    % Torque = I*alpha + omega x (I*omega)
    Tau_ang = (I * alpha(frame,:)' + cross(omega(frame,:)', I*omega(frame,:)'))';
    Tau_force = cross(r, F);  % Moment due to force
    Tau = Tau_ang - Tau_force;
end
