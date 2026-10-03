function tests = test_gs3dx_runtime_fingerprint
% Pure unit tests for gs3dx_runtime_fingerprint: missing components, privacy, identity.
    tests = functiontests(localfunctions);
end

function setupOnce(t)
    t.TestData.original_path = path;
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools'));
end

function teardownOnce(t)
    path(t.TestData.original_path);
end

function s = helperMockVer()
    s = [ ...
        struct('Name', 'Simulink', 'Version', '25.2', 'Release', 'R2025b'); ...
        struct('Name', 'Simscape', 'Version', '25.2', 'Release', 'R2025b'); ...
        struct('Name', 'Simscape Multibody', 'Version', '25.2', 'Release', 'R2025b') ...
    ];
end

function testValidRuntimeFingerprint(t)
    v_mat = '25.2.0.2790938';
    r_mat = 'R2025b';
    all_ver = helperMockVer();
    py_ver = '3.11.9';
    np_ver = '1.26.4';
    ez_ver = '1.5.10';

    fp = gs3dx_runtime_fingerprint(v_mat, r_mat, all_ver, py_ver, np_ver, ez_ver);

    verifyEqual(t, fp.matlab_release, 'R2025b');
    verifyEqual(t, fp.matlab_version, '25.2.0.2790938');
    verifyEqual(t, fp.matlab.version, '25.2.0.2790938');
    verifyEqual(t, fp.matlab.release, 'R2025b');
    verifyEqual(t, fp.simulink.version, '25.2');
    verifyEqual(t, fp.simscape.version, '25.2');
    verifyEqual(t, fp.simscape_multibody.version, '25.2');
    verifyEqual(t, fp.python_version, '3.11.9');
    verifyEqual(t, fp.numpy_version, '1.26.4');
    verifyEqual(t, fp.ezc3d_version, '1.5.10');
    verifyEqual(t, fp.scope, 'direct_components_recorded');
    verifyTrue(t, contains(fp.description, 'transitive packages are not implied to be frozen'));
end

function testMatlabReleaseNormalization(t)
    fp = gs3dx_runtime_fingerprint('25.2.0', '2025b', helperMockVer(), '3.11.9', '1.26.4', '1.5.10');
    verifyEqual(t, fp.matlab_release, 'R2025b');
    verifyEqual(t, fp.matlab.release, 'R2025b');
end

function testMissingMatlabVersionFailsClosed(t)
    verifyError(t, @() gs3dx_runtime_fingerprint('', 'R2025b', helperMockVer(), '3.11.9', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testMissingMatlabReleaseFailsClosed(t)
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', '', helperMockVer(), '3.11.9', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testMissingSimulinkFailsClosed(t)
    no_simulink = [ ...
        struct('Name', 'Simscape', 'Version', '25.2', 'Release', 'R2025b'); ...
        struct('Name', 'Simscape Multibody', 'Version', '25.2', 'Release', 'R2025b') ...
    ];
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', no_simulink, '3.11.9', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testMissingSimscapeFailsClosed(t)
    no_simscape = [ ...
        struct('Name', 'Simulink', 'Version', '25.2', 'Release', 'R2025b'); ...
        struct('Name', 'Simscape Multibody', 'Version', '25.2', 'Release', 'R2025b') ...
    ];
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', no_simscape, '3.11.9', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testMissingSimscapeMultibodyFailsClosed(t)
    no_sm = [ ...
        struct('Name', 'Simulink', 'Version', '25.2', 'Release', 'R2025b'); ...
        struct('Name', 'Simscape', 'Version', '25.2', 'Release', 'R2025b') ...
    ];
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', no_sm, '3.11.9', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testExactActualNameMatchingRequired(t)
    inexact_ver = [ ...
        struct('Name', 'Simulink', 'Version', '25.2', 'Release', 'R2025b'); ...
        struct('Name', 'Simscape', 'Version', '25.2', 'Release', 'R2025b'); ...
        struct('Name', 'Simscape Multibody Toolbox', 'Version', '25.2', 'Release', 'R2025b') ...
    ];
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', inexact_ver, '3.11.9', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testEmptyProductVersionFailsClosed(t)
    empty_ver = [ ...
        struct('Name', 'Simulink', 'Version', '', 'Release', 'R2025b'); ...
        struct('Name', 'Simscape', 'Version', '25.2', 'Release', 'R2025b'); ...
        struct('Name', 'Simscape Multibody', 'Version', '25.2', 'Release', 'R2025b') ...
    ];
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', empty_ver, '3.11.9', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testMissingPythonVersionFailsClosed(t)
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', helperMockVer(), '', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testMissingNumpyVersionFailsClosed(t)
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', helperMockVer(), '3.11.9', '', '1.5.10'), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testMissingEzc3dVersionFailsClosed(t)
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', helperMockVer(), '3.11.9', '1.26.4', ''), ...
        'gs3dx:match_export:MissingRuntimeDependency');
end

function testPrivacyViolationAbsoluteWindowsPath(t)
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', helperMockVer(), 'C:\Python311\python.exe', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:PrivacyViolation');
end

function testPrivacyViolationAbsolutePosixPath(t)
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', helperMockVer(), '/usr/bin/python3', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:PrivacyViolation');
end

function testPrivacyViolationUserDirectory(t)
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', helperMockVer(), '/home/user/python', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:PrivacyViolation');
end

function testPrivacyViolationSecretKeyword(t)
    verifyError(t, @() gs3dx_runtime_fingerprint('25.2.0', 'R2025b', helperMockVer(), 'token_api_secret', '1.26.4', '1.5.10'), ...
        'gs3dx:match_export:PrivacyViolation');
end

function testStructArgumentFormSupported(t)
    s = struct( ...
        'matlab_version', '25.2.0.2790938', ...
        'matlab_release', 'R2025b', ...
        'ver', helperMockVer(), ...
        'python_version', '3.11.9', ...
        'numpy_version', '1.26.4', ...
        'ezc3d_version', '1.5.10');
    fp = gs3dx_runtime_fingerprint(s);
    verifyEqual(t, fp.matlab_release, 'R2025b');
    verifyEqual(t, fp.python_version, '3.11.9');
end
