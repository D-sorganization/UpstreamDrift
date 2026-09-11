classdef test_project_body_markers < matlab.unittest.TestCase
    methods (Test)
        function offsetsRotateWithAssignedBodies(testCase)
            rotations = cat(3, eye(3), [0 -1 0;1 0 0;0 0 1]);
            actual = project_body_markers([10 20 30;1 2 3], rotations, ...
                [2;1;2], [1 0 0;0 0 1;0 1 0]);
            testCase.verifyEqual(actual, [1 3 3;10 20 31;0 2 3]);
        end
        function rejectsUnknownBody(testCase)
            testCase.verifyError(@() project_body_markers(zeros(1,3),eye(3),2,zeros(1,3)), ...
                'project_body_markers:bodyIndex');
        end
        function rejectsReflectedOrScaledFrames(testCase)
            for rotation = {diag([1 1 -1]), 2*eye(3)}
                testCase.verifyError(@() project_body_markers(zeros(1,3),rotation{1},1,zeros(1,3)), ...
                    'project_body_markers:rotation');
            end
        end
        function rejectsMismatchedAttachments(testCase)
            testCase.verifyError(@() project_body_markers(zeros(1,3),eye(3),[1;1],zeros(1,3)), ...
                'project_body_markers:attachmentCount');
        end
    end
end
