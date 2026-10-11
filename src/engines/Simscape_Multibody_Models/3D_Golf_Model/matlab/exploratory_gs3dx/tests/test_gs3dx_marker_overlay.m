function tests = test_gs3dx_marker_overlay
    tests = functiontests(localfunctions);
end
function setupOnce(t)
    t.TestData.original_path = path;
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools'));
end
function teardownOnce(t)
    path(t.TestData.original_path);
end
function testCaptureFramesAlignToPoseColumns(t)
    names = ["pelvis", "c7", "shoulder_L", "shoulder_R", ...
        "elbow_L", "elbow_R", "wrist_L", "wrist_R", ...
        "hip_L", "hip_R", "knee_L", "knee_R", ...
        "ankle_L", "ankle_R", "toe_L", "toe_R", "club_grip", "club_head"];
    jc = struct();
    for k = 1:numel(names)
        jc.(names(k)) = [1:20; (1:20)+100; (1:20)+200] + k*1000;
    end
    out = gs3dx_marker_overlay(jc, [1 9 17]);
    verifySize(t, out, [3 18 3]);
    verifyEqual(t, squeeze(out(:, 18, :)), jc.club_head(:, [1 9 17]));
    verifyEqual(t, squeeze(out(:, 1, :)), jc.pelvis(:, [1 9 17]));
end
