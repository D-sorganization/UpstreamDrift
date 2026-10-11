function tests = test_gs3dx_match_export
% Pure precondition and public API checks: no Simulink compilation or capture access.
    tests = functiontests(localfunctions);
end

function setupOnce(t)
    t.TestData.original_path = path;
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools'));
end

function teardownOnce(t)
    path(t.TestData.original_path);
end

function testNoInventedDynamics(t)
    dest = tempname;
    verifyError(t, @() gs3dx_match_export("capture-A", mode="tracked", output_dir=dest), ...
        'gs3dx:match_export:UnsupportedMode');
    verifyFalse(t, isfolder(dest));
end

function testNoUnverifiedPoseSubstitution(t)
    dest = tempname;
    verifyError(t, @() gs3dx_match_export("capture-O", pose_source=struct('joint', zeros(3)), output_dir=dest), ...
        'gs3dx:match_export:UnverifiedPoses');
    verifyFalse(t, isfolder(dest));
end

function testUnsupportedTopologyRejected(t)
    % GS3DX_Human is required for all new match exports; legacy baseline GS3DX_Fit is rejected
    dest = tempname;
    unsupported = ["GS3DX_Fit", "GS3DX_Neck", "GS3DX_Shape", "GS3DX_Golfer", "invented_model"];
    for mdl = unsupported
        verifyError(t, @() gs3dx_match_export("capture-O", model=mdl, output_dir=dest), ...
            'gs3dx:match_export:UnsupportedModel');
    end
    verifyFalse(t, isfolder(dest));
end

function testHumanModelAccepted(t)
    % GS3DX_Human passes model option validation and reaches capture resolution
    dest = tempname;
    missing_capture = string(tempname) + ".c3d";
    verifyError(t, @() gs3dx_match_export(missing_capture, model="GS3DX_Human", output_dir=dest), ...
        'gs3dx:match_export:CaptureNotFound');
    verifyFalse(t, isfolder(dest));
end

function testDefaultModelIsAccepted(t)
    % Omitting model defaults to GS3DX_Human and reaches capture resolution
    dest = tempname;
    missing_capture = string(tempname) + ".c3d";
    verifyError(t, @() gs3dx_match_export(missing_capture, output_dir=dest), ...
        'gs3dx:match_export:CaptureNotFound');
    verifyFalse(t, isfolder(dest));
end

function testEmptyCaptureIdRejected(t)
    verifyError(t, @() gs3dx_match_export("", output_dir=tempname), ...
        'gs3dx:match_export:EmptyCaptureId');
end

function testOutputMustBeExplicit(t)
    verifyError(t, @() gs3dx_match_export("capture-A"), ...
        'gs3dx:match_export:MissingOutputDir');
end

function testInvalidStride(t)
    for stride = [0.2, Inf, NaN, -1]
        verifyError(t, @() gs3dx_match_export("capture-A", stride=stride, output_dir=tempname), ...
            'gs3dx:match_export:InvalidStride');
    end
end

function testInvalidFrames(t)
    verifyError(t, @() gs3dx_match_export("capture-A", frames=[1 Inf], output_dir=tempname), ...
        'gs3dx:match_export:InvalidFrames');
    verifyError(t, @() gs3dx_match_export("capture-A", frames=[10 5], output_dir=tempname), ...
        'gs3dx:match_export:InvalidFrames');
    verifyError(t, @() gs3dx_match_export("capture-A", frames=[-1 5], output_dir=tempname), ...
        'gs3dx:match_export:InvalidFrames');
end

function testInvalidView(t)
    verifyError(t, @() gs3dx_match_export("capture-A", views="invented", output_dir=tempname), ...
        'gs3dx:match_export:InvalidViews');
    verifyError(t, @() gs3dx_match_export("capture-A", views=strings(0,0), output_dir=tempname), ...
        'gs3dx:match_export:InvalidViews');
end

function testInvalidStills(t)
    verifyError(t, @() gs3dx_match_export("capture-A", stills=NaN, output_dir=tempname), ...
        'gs3dx:match_export:InvalidStills');
    verifyError(t, @() gs3dx_match_export("capture-A", stills=0, output_dir=tempname), ...
        'gs3dx:match_export:InvalidStills');
    verifyError(t, @() gs3dx_match_export("capture-A", stills=1.5, output_dir=tempname), ...
        'gs3dx:match_export:InvalidStills');
end

function testNonExistentC3DFileFailsClosed(t)
    dest = tempname;
    bad_c3d = fullfile(tempdir, "nonexistent_capture_file_987654.c3d");
    verifyError(t, @() gs3dx_match_export(bad_c3d, output_dir=dest), ...
        'gs3dx:match_export:CaptureNotFound');
    verifyFalse(t, isfolder(dest));
end
