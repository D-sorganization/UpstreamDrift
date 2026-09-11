classdef test_extract_golf_joint_kinematics < matlab.unittest.TestCase
    methods (Test)
        function convertsNativeDegreesAndPreservesJointOrder(testCase)
            logs = struct;
            logs.LELogs.LEJoint.AngularPosition = timeseries([0; 90; 180], [0; 0.1; 1]);
            logs.HipLogs.HipAngularPositionZ = timeseries([180; 180], [0; 1]);
            logs.LELogs.LEJoint.AngularVelocity = timeseries([90; 180], [0; 1]);
            logs.LELogs.LEJoint.AngularAcceleration = timeseries([180; 360], [0; 1]);
            result = extract_golf_joint_kinematics(logs, ...
                ["HipInputZ", "LEInput"], [0; 0.1; 1]);
            testCase.verifyEqual(result.q(:,1), pi * ones(3,1), 'AbsTol', 1e-12);
            testCase.verifyEqual(result.q(:,2), [0; pi/2; pi], 'AbsTol', 1e-12);
            testCase.verifyEqual(result.qd(end,2), pi, 'AbsTol', 1e-12);
            testCase.verifyEqual(result.qdd(end,2), 2*pi, 'AbsTol', 1e-12);
            testCase.verifyEqual(result.coordinate_units, ["rad", "rad"]);
        end

        function translationsRemainMetres(testCase)
            logs.HipLogs.HipPositionY = timeseries([1; 2], [0; 1]);
            logs.HipLogs.HipVelocityY = timeseries([3; 4], [0; 1]);
            result = extract_golf_joint_kinematics(logs, "TranslationInputY", [0; 1]);
            testCase.verifyEqual(result.q, [1; 2]);
            testCase.verifyEqual(result.qd, [3; 4]);
            testCase.verifyEqual(result.coordinate_units, "m");
        end

        function missingSignalsStayMissing(testCase)
            result = extract_golf_joint_kinematics(struct, "LSInputZ", [0; 1]);
            testCase.verifyTrue(all(isnan([result.q; result.qd; result.qdd])));
            testCase.verifyEqual(result.source_names.q, "LSLogs.AngularPosition_Z");
        end

        function usesScalarSensorRatherThanMiswiredAggregate(testCase)
            logs.AngularKinematicsLogs.RScapAngularAccelerationX = timeseries(ones(2,3), [0; 1]);
            logs.RScapLogs.AngularAccelerationX = timeseries([90; 180], [0; 1]);
            logs.LSLogs.AngularPosition_Z = timeseries([180; 180], [0; 1]);
            result = extract_golf_joint_kinematics(logs, ...
                ["RScapInputX", "LSInputZ"], [0; 1]);
            testCase.verifyEqual(result.qdd(:,1), [pi/2; pi], 'AbsTol', 1e-12);
            testCase.verifyEqual(result.q(:,2), [pi; pi], 'AbsTol', 1e-12);
        end

        function rejectsUnknownCoordinates(testCase)
            testCase.verifyError(@() extract_golf_joint_kinematics(struct, ...
                "LEInputX", [0; 1]), 'extract_golf_joint_kinematics:unknownJoint');
        end
    end
end
