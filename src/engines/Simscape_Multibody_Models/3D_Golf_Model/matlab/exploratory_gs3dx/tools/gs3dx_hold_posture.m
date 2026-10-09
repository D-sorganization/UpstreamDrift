function vars = gs3dx_hold_posture(start, opts)
%GS3DX_HOLD_POSTURE  Model-workspace overrides that hold the upper body at its start pose (#11709).
%
%   VARS = GS3DX_HOLD_POSTURE(START) returns the workspace overrides that
%   switch every upper-body '<prefix> Input Function' chart
%   (GS3DX_UPPER_BODY_JOINTS) to its existing ModelingMode 2 law,
%
%     tau = Killswitch*GlobalGain*(Constant + TimeGain*t
%           - PositionGain*(q - PositionSetpoint) - Dampening*qdot),
%
%   with Constant = TimeGain = 0, PositionSetpoint = the joint's start
%   angle, GlobalGain = 1 and the killswitch held on.  That is a joint-space
%   spring-damper about the start pose: the standing test for ground
%   reactions, where the driven upper body would otherwise collapse.  No
%   block or saved model changes; GS3DX_CONTACT_CHECK applies VARS through
%   its VARIABLES option.
%
%   START is a struct (or containers.Map-like getter result) holding every
%   '<prefix>StartPosition<axis>' variable, in degrees, e.g. the drive's
%   overrides merged over the model workspace.
%
%   Options (N*m/deg and N*m*s/deg):
%     trunk     stiffness of Spine and Torso (40)
%     shoulder  stiffness of LS, RS, LScap, RScap (15)
%     arm       stiffness of LE, RE, LF, RF, LW, RW (5)
%     damping_ratio  Dampening = stiffness * damping_ratio (0.1 s)
%
%   Postconditions: VARS.ModelingMode == 2 and, for every joint axis,
%   VARS.<P>PositionSetpoint<ax> == START.<P>StartPosition<ax> and
%   VARS.<P>PositionGain<ax> > 0.

    arguments
        start (1,1) struct
        opts.trunk (1,1) double {mustBePositive} = 40
        opts.shoulder (1,1) double {mustBePositive} = 15
        opts.arm (1,1) double {mustBePositive} = 5
        opts.damping_ratio (1,1) double {mustBeNonnegative} = 0.1
    end
    groups = struct('trunk', {{'Spine', 'Torso'}}, ...
        'shoulder', {{'LS', 'RS', 'LScap', 'RScap'}}, ...
        'arm', {{'LE', 'RE', 'LF', 'RF', 'LW', 'RW'}});
    vars = struct('ModelingMode', 2, 'GlobalGain', 1, ...
        'KillswitchInitialValue', 1, 'KillswitchFinalValue', 1);
    for j = reshape(gs3dx_upper_body_joints(), 1, [])
        k = local_stiffness(j.prefix, groups, opts);
        axes = num2cell(j.axes);
        if isempty(axes)
            axes = {''};
        end
        for a = axes
            ax = a{1};
            name = [j.prefix 'StartPosition' ax];
            assert(isfield(start, name), 'gs3dx:hold_posture', ...
                'Precondition: START has no %s', name);
            vars.([j.prefix 'PositionSetpoint' ax]) = local_value(start.(name));
            vars.([j.prefix 'PositionGain' ax]) = k;
            vars.([j.prefix 'Dampening' ax]) = k * opts.damping_ratio;
            vars.([j.prefix 'Constant' ax]) = 0;
            vars.([j.prefix 'TimeGain' ax]) = 0;
        end
    end
end

function k = local_stiffness(prefix, groups, opts)
    for g = reshape(fieldnames(groups), 1, [])
        if any(strcmp(prefix, groups.(g{1})))
            k = opts.(g{1});
            return;
        end
    end
    error('gs3dx:hold_posture', 'Joint %s belongs to no stiffness group', prefix);
end

function v = local_value(v)
    if isa(v, 'Simulink.Parameter')
        v = v.Value;
    end
    v = double(v);
end
