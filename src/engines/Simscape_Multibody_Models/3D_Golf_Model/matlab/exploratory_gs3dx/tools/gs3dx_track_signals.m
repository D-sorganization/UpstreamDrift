function sig = gs3dx_track_signals(prefix, axes)
%GS3DX_TRACK_SIGNALS  Chart signal names of an upper-body joint (#10979).
%
%   SIG = GS3DX_TRACK_SIGNALS(PREFIX, AXES) names the '<PREFIX> Input
%   Function' chart data of a joint with AXES ('' for one angle, 'XY',
%   'XYZ'): SIG.q and SIG.qd (angle and rate inputs, deg and deg/s) and
%   SIG.tau (torque outputs, N*m), cell rows in axis order.

    arguments
        prefix (1,:) char
        axes (1,:) char {mustBeMember(axes, {'', 'XY', 'XYZ'})} = ''
    end
    if isempty(axes)
        sig = struct('q', {{[prefix 'Position']}}, 'qd', {{[prefix 'Velocity']}}, ...
            'tau', {{['JointTorque' prefix]}});
        return
    end
    a = num2cell(axes);
    sig = struct('q', {strcat(prefix, 'Position', a)}, 'qd', {strcat(prefix, 'Velocity', a)}, ...
        'tau', {strcat('JointTorque', prefix, a)});
end
