classdef test_audit_golf_actuator_torques < matlab.unittest.TestCase
    methods (Test)
        function comparesPolynomialOnNativeClock(testCase)
            clock=[0;0.07;0.2];
            bus.LSLogs.ActuatorTorqueX=timeseries(3+5*clock,clock);
            theta=zeros(14,1);theta(6)=5;theta(7)=3;
            actual=audit_golf_actuator_torques(bus,theta,["LSInputX","TorsoInput"]);
            testCase.verifyEqual(actual.entries.max_abs_error,0,'AbsTol',1e-14);
            testCase.verifyEqual(actual.entries.sample_count,3);
            testCase.verifyEqual(actual.unlogged_coordinates,"TorsoInput");
            bus.LSLogs.ActuatorTorqueX.Data(2)=bus.LSLogs.ActuatorTorqueX.Data(2)+0.4;
            changed=audit_golf_actuator_torques(bus,theta,["LSInputX","TorsoInput"]);
            testCase.verifyEqual(changed.entries.max_abs_error,0.4,'AbsTol',1e-14);
        end
        function rootEffortsHaveSeparateForceAndTorqueUnits(testCase)
            clock=[0;0.03;0.2];rotation=[1 0 0;0 0 1;0 -1 0];
            world=[1+2*clock,10+3*clock,700+50*clock];local=world*rotation';
            for j=1:3
                axis=char('X'+j-1);
                bus.HipLogs.("TranslationForce"+string(axis)+"Input")=timeseries(local(:,j),clock);
            end
            bus.HipLogs.HipTorqueYInput=timeseries(-30+5*clock,clock);
            theta=zeros(7,4);theta(6,:)=[2 3 50 5];theta(7,:)=[1 10 700 -30];
            names=["TranslationInputX","TranslationInputY","TranslationInputZ","HipInputY"];
            actual=audit_golf_actuator_torques(bus,theta(:),names,rotation);
            testCase.verifyEqual([actual.entries.unit],["N","N","N","Nm"]);
            testCase.verifyEqual(actual.max_force_error_N,0,'AbsTol',1e-12);
            testCase.verifyEqual(actual.max_torque_error_Nm,0,'AbsTol',1e-12);
            testCase.verifyEmpty(actual.unlogged_coordinates);
            bus.HipLogs.TranslationForceYInput.Data(2)=local(2,2)+0.5;
            changed=audit_golf_actuator_torques(bus,theta(:),names,rotation);
            testCase.verifyEqual(changed.max_force_error_N,0.5,'AbsTol',1e-12);
            testCase.verifyEqual(changed.max_torque_error_Nm,0,'AbsTol',1e-12);
            testCase.verifyError(@()audit_golf_actuator_torques(bus,theta(:),names), ...
                'audit_golf_actuator_torques:forceFrame');
            testCase.verifyError(@()audit_golf_actuator_torques(bus,theta(:),names,zeros(3)), ...
                'audit_golf_actuator_torques:forceFrame');
        end
        function auditsNewRevoluteSensorsWhenPresent(testCase)
            clock=[0;0.07;0.3];signal=timeseries(2+3*clock,clock);
            bus.TorsoLogs.ActuatorTorque=signal;
            bus.LELogs.LEJoint.ActuatorTorque=signal;
            bus.LFLogs.ActuatorTorque=signal;
            bus.RELogs.REJoint.ActuatorTorque=signal;
            bus.RFLogs.ActuatorTorque=signal;
            names=["TorsoInput","LEInput","LFInput","REInput","RFInput"];
            theta=zeros(7,5);theta(6,:)=3;theta(7,:)=2;
            result=audit_golf_actuator_torques(bus,theta(:),names);
            testCase.verifyNumElements(result.entries,5);
            testCase.verifyEmpty(result.unlogged_coordinates);
            testCase.verifyEqual(result.max_torque_error_Nm,0,'AbsTol',1e-12);
            bus.LFLogs.ActuatorTorque=timeseries(2.25+3*clock,clock);
            result=audit_golf_actuator_torques(bus,theta(:),names);
            testCase.verifyEqual(result.max_torque_error_Nm,0.25,'AbsTol',1e-12);
            bus.LFLogs=rmfield(bus.LFLogs,'ActuatorTorque');
            legacy=audit_golf_actuator_torques(bus,theta(:),names);
            testCase.verifyEqual(legacy.unlogged_coordinates,"LFInput");
        end
        function missingActuatorIsNotReplacedByReaction(testCase)
            bus.LSLogs.TorqueLocal=timeseries(ones(3,3),[0;0.1;0.2]);
            testCase.verifyError(@()audit_golf_actuator_torques(bus,zeros(7,1),"LSInputX"), ...
                'audit_golf_actuator_torques:missingLog');
        end
    end
end
