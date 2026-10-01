function g = gs3dx_track_gains(opts)
%GS3DX_TRACK_GAINS  Default PD gains of the upper-body capture tracking (#10979).
%
%   G = GS3DX_TRACK_GAINS() is a struct of chart prefix -> [Kp Kd], one row
%   per joint axis, in N*m/deg and N*m/(deg/s).  Each joint is a critically
%   damped servo of natural frequency FREQ_HZ on the inertia it carries:
%       Kp = I w^2,  Kd = 2 ZETA I w   (w = 2 pi FREQ_HZ, per radian)
%   converted to degrees.  I (kg*m^2) is a rough estimate for an 80 kg
%   de Leva body with the club: what distal to each joint rotates about it.
%   Uniform stiffness instead drives the light distal joints (forearm,
%   wrist) far above their bandwidth, and the solver stalls on them.  The PD
%   corrects only what the learned feedforward (GS3DX_TRACK_LEARN) leaves.
%
%   Options: freq_hz (6), zeta (1).

    arguments
        opts.freq_hz (1,1) double {mustBePositive} = 6
        opts.zeta (1,1) double {mustBePositive} = 1
    end
    inertia = struct('Spine', 3, 'Torso', 2, 'LScap', 0.6, 'RScap', 0.6, 'LS', 0.5, 'RS', 0.5, ...
        'LE', 0.15, 'RE', 0.15, 'LF', 0.02, 'RF', 0.02, 'LW', 0.3, 'RW', 0.3);
    axes = struct('Spine', 2, 'Torso', 1, 'LScap', 2, 'RScap', 2, 'LS', 3, 'RS', 3, ...
        'LE', 1, 'RE', 1, 'LF', 1, 'RF', 1, 'LW', 2, 'RW', 2);
    w = 2 * pi * opts.freq_hz;
    g = struct();
    for f = fieldnames(inertia).'
        I = inertia.(f{1});
        g.(f{1}) = repmat(deg2rad([I * w ^ 2, 2 * opts.zeta * I * w]), axes.(f{1}), 1);
    end
end
