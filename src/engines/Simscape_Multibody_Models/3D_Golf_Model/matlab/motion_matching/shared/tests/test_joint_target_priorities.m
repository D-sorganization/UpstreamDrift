classdef test_joint_target_priorities < matlab.unittest.TestCase
    methods (Test)
        function gimbalInstancesRespectIndependentPriorities(testCase)
            [host, first, second] = testCase.instances('Gimbal');
            set_param(first, 'RxPositionTargetPriority', 'Low');
            set_param(second, 'RxPositionTargetPriority', 'High');
            testCase.verifyEqual(get_param([first '/Kinetically Driven'], ...
                'RxPositionTargetPriority'), 'Low');
            testCase.verifyEqual(get_param([second '/Kinetically Driven'], ...
                'RxPositionTargetPriority'), 'High');
            set_param(first, 'RxPositionTargetPriority', 'None');
            testCase.verifyEqual(get_param([first '/Kinetically Driven'], ...
                'RxPositionTargetSpecify'), 'off');
            set_param(first, 'RxPositionTargetPriority', 'High');
            testCase.verifyEqual(get_param([first '/Kinetically Driven'], ...
                'RxPositionTargetSpecify'), 'on');
            testCase.verifyTrue(bdIsLoaded(host));
        end

        function everySelectorReachesItsPrimitive(testCase)
            kinds = {'Gimbal', 'Universal', 'Revolute'};
            primitives = {'Kinetically Driven', 'Kinetically Driven Universal Joint', ...
                'Kinetically Driven Revolute'};
            axes = {{'Rx','Ry','Rz'}, {'Rx','Ry'}, {'Rz'}};
            for k = 1:numel(kinds)
                [~, first] = testCase.instances(kinds{k});
                primitive = [first '/' primitives{k}];
                for axis = axes{k}
                    target_axis = axis{1};
                    if strcmp(kinds{k}, 'Revolute'), target_axis = ''; end
                    for quantity = {'Position','Velocity'}
                        selector = [axis{1} quantity{1} 'TargetPriority'];
                        prefix = [target_axis quantity{1} 'Target'];
                        for selection = {'None','Low','High'}
                            set_param(first, selector, selection{1});
                            if strcmp(selection{1}, 'None')
                                testCase.verifyEqual(get_param(primitive, [prefix 'Specify']), 'off');
                            else
                                testCase.verifyEqual(get_param(primitive, [prefix 'Specify']), 'on');
                                testCase.verifyEqual(get_param(primitive, [prefix 'Priority']), selection{1});
                            end
                        end
                    end
                end
            end
        end
    end

    methods (Access = private)
        function [host, first, second] = instances(testCase, kind)
            directory = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'src', 'model');
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(directory));
            reference = ['Kinetically_Driven_' kind '_Joint'];
            testCase.addTeardown(@() bdclose(reference));
            [~, host] = fileparts(tempname);
            new_system(host);
            testCase.addTeardown(@() bdclose(host));
            workspace = get_param(host, 'ModelWorkspace');
            assignin(workspace, 'LocalDampeningEnable', 0);
            assignin(workspace, 'DampeningGlobalGain', 1);
            first = [host '/First'];
            second = [host '/Second'];
            add_block('built-in/SubSystem', first, 'ReferencedSubsystem', reference);
            add_block('built-in/SubSystem', second, 'ReferencedSubsystem', reference);
        end
    end
end
