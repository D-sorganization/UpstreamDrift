classdef test_revolute_actuator_log < matlab.unittest.TestCase
    methods (Test)
        function addsSensedScalarWithoutReplacingExistingSignals(testCase)
            model='Kinetically_Driven_Revolute_Joint';
            testCase.assertFalse(bdIsLoaded(model));load_system(model);
            cleanup=onCleanup(@()close_system(model,0)); %#ok<NASGU>
            bus=find_system(model,'SearchDepth',1,'BlockType','BusCreator');
            testCase.assertNumElements(bus,1);
            converter=[model '/ActuatorTorque'];
            if getSimulinkBlockHandle(converter)>0
                handles=get_param(converter,'PortHandles');
                delete_line(get_param(handles.LConn(1),'Line'));
                delete_line(get_param(handles.Outport(1),'Line'));
                delete_block(converter);
                set_param(bus{1},'Inputs','12');
                set_param([model '/Kinetically Driven Revolute'],'SenseTorqueForce','off');
            end
            ports=get_param(bus{1},'PortHandles');
            before=arrayfun(@(p)string(get_param(get_param(p,'Line'),'Name')),ports.Inport);
            enable_revolute_actuator_log(model);
            testCase.verifyEqual(get_param([model '/Kinetically Driven Revolute'],'SenseTorqueForce'),'on');
            ports=get_param(bus{1},'PortHandles');
            after=arrayfun(@(p)string(get_param(get_param(p,'Line'),'Name')),ports.Inport);
            testCase.verifyEqual(after(1:numel(before)),before);
            testCase.verifyEqual(after(end),"ActuatorTorque");
            testCase.verifyEqual(numel(after),numel(before)+1);
            converter=[model '/ActuatorTorque'];
            handles=get_param(converter,'PortHandles');
            testCase.verifyNotEqual(get_param(handles.LConn(1),'Line'),-1);
            testCase.verifyEqual(get_param(converter,'Unit'),'N*m');
            enable_revolute_actuator_log(model);
            testCase.verifyEqual(str2double(get_param(bus{1},'Inputs')),numel(after));
        end
    end
end
