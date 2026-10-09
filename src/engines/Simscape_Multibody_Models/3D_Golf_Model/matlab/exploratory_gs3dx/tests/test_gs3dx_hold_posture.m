classdef test_gs3dx_hold_posture < matlab.unittest.TestCase
    % The held-stance overrides (#11709): a ModelingMode 2 spring-damper
    % about each upper-body joint's start angle, nothing else.
    methods (TestClassSetup)
        function setup(~)
            gs3dx_setup();
        end
    end
    methods (Test)
        function every_joint_axis_is_held_at_its_start_angle(t)
            [start, names] = local_start();
            v = gs3dx_hold_posture(start);
            t.verifyEqual(v.ModelingMode, 2);
            t.verifyEqual([v.GlobalGain, v.KillswitchInitialValue, v.KillswitchFinalValue], [1 1 1]);
            for k = 1:numel(names)
                [P, ax] = deal(names{k}{:});
                t.verifyEqual(v.([P 'PositionSetpoint' ax]), start.([P 'StartPosition' ax]));
                t.verifyGreaterThan(v.([P 'PositionGain' ax]), 0);
                t.verifyEqual(v.([P 'Constant' ax]), 0);
                t.verifyEqual(v.([P 'TimeGain' ax]), 0);
            end
            t.verifyEqual(numel(names), 21);   % 6 hinges + 4*2 universal + 2*3 shoulder axes
        end
        function gains_follow_the_groups(t)
            v = gs3dx_hold_posture(local_start(), trunk=40, shoulder=15, arm=5, damping_ratio=0.1);
            t.verifyEqual(v.SpinePositionGainX, 40);
            t.verifyEqual(v.TorsoPositionGain, 40);
            t.verifyEqual(v.LSPositionGainZ, 15);
            t.verifyEqual(v.RScapPositionGainY, 15);
            t.verifyEqual(v.LEPositionGain, 5);
            t.verifyEqual(v.RWDampeningY, 0.5, 'AbsTol', 1e-12);
        end
        function a_missing_start_angle_is_refused(t)
            start = rmfield(local_start(), 'LEStartPosition');
            t.verifyError(@() gs3dx_hold_posture(start), 'gs3dx:hold_posture');
        end
        function only_documented_variables_are_overridden(t)
            v = gs3dx_hold_posture(local_start());
            ok = ismember(fieldnames(v), {'ModelingMode', 'GlobalGain', ...
                'KillswitchInitialValue', 'KillswitchFinalValue'}) | ...
                ~cellfun('isempty', regexp(fieldnames(v), ...
                '(PositionSetpoint|PositionGain|Dampening|Constant|TimeGain)[XYZ]?$', 'once'));
            t.verifyTrue(all(ok));
        end
    end
end

function [start, names] = local_start()
    start = struct();
    names = {};
    for j = reshape(gs3dx_upper_body_joints(), 1, [])
        axes = num2cell(j.axes);
        if isempty(axes)
            axes = {''};
        end
        for a = axes
            start.([j.prefix 'StartPosition' a{1}]) = 10 * numel(names) - 7;
            names{end + 1} = {j.prefix, a{1}}; %#ok<AGROW>
        end
    end
end
