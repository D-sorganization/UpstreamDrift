function tests = test_gs3dx_video_sampling
    tests = functiontests(localfunctions);
end
function setupOnce(t)
    t.TestData.path = path;
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools'));
end
function teardownOnce(t)
    path(t.TestData.path);
end
function testPhysicalTiming(t)
    [frames, stride, fps] = gs3dx_video_sampling(360, 654, 12, []);
    verifyEqual(t, frames, 1:12:654);
    verifyEqual(t, stride, 12);
    verifyEqual(t, fps, 30);
    verifyEqual(t, (frames(end)-frames(1))/360, (numel(frames)-1)/fps);
end
function testIrregularFramesRejected(t)
    verifyError(t, @() gs3dx_video_sampling(360, 654, 0, [1 13 654]), 'gs3dx:video_sampling');
end
function testFractionalStrideRejected(t)
    verifyError(t, @() gs3dx_video_sampling(360, 654, 0.2, []), 'gs3dx:video_sampling');
end
function testOutsideCaptureRejected(t)
    verifyError(t, @() gs3dx_video_sampling(360, 654, 0, [1 655]), 'gs3dx:video_sampling');
end
function testInvalidRateRejected(t)
    verifyError(t, @() gs3dx_video_sampling(NaN, 654, 12, []), 'gs3dx:video_sampling');
end
