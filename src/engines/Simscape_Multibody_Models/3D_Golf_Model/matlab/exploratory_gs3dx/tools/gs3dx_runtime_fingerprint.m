function fp = gs3dx_runtime_fingerprint(v_mat, r_mat, all_ver, py_ver, np_ver, ez_ver)
%GS3DX_RUNTIME_FINGERPRINT Pure runtime component fingerprint validation (#10979, #11161).
%   FP = GS3DX_RUNTIME_FINGERPRINT(V_MAT, R_MAT, ALL_VER, PY_VER, NP_VER, EZ_VER)
%   accepts actual runtime values:
%     V_MAT    - MATLAB version string (e.g. '25.2.0.2790938')
%     R_MAT    - MATLAB release string (e.g. 'R2025b')
%     ALL_VER  - struct array returned by ver (exact product names)
%     PY_VER   - Python version string with major.minor.micro (from sys.version_info)
%     NP_VER   - numpy version string
%     EZ_VER   - ezc3d version string
%
%   Fails closed on missing or empty values, missing MathWorks toolboxes, or
%   privacy violations (path leaks, license/token keywords).
%
%   Describes recorded direct execution components only; does not claim that
%   unrecorded transitive dependencies or system packages are frozen.

    if nargin == 1 && isstruct(v_mat)
        s = v_mat;
        assert(isfield(s, 'matlab_version') && isfield(s, 'matlab_release') && ...
               isfield(s, 'ver') && isfield(s, 'python_version') && ...
               isfield(s, 'numpy_version') && isfield(s, 'ezc3d_version'), ...
               'gs3dx:match_export:MissingRuntimeDependency', ...
               'Struct input missing required runtime fields.');
        v_mat = s.matlab_version;
        r_mat = s.matlab_release;
        all_ver = s.ver;
        py_ver = s.python_version;
        np_ver = s.numpy_version;
        ez_ver = s.ezc3d_version;
    elseif nargin ~= 6
        error('gs3dx:match_export:MissingRuntimeDependency', ...
            'gs3dx_runtime_fingerprint requires 6 arguments: (matlab_version, matlab_release, ver_struct, python_version, numpy_version, ezc3d_version)');
    end

    % 1. MATLAB release and version
    assert(~isempty(v_mat) && strlength(string(v_mat)) > 0, ...
        'gs3dx:match_export:MissingRuntimeDependency', 'MATLAB version could not be determined.');
    assert(~isempty(r_mat) && strlength(string(r_mat)) > 0, ...
        'gs3dx:match_export:MissingRuntimeDependency', 'MATLAB release could not be determined.');
    r_mat = char(r_mat);
    if ~startsWith(r_mat, 'R')
        r_mat = ['R' r_mat];
    end
    v_mat = char(v_mat);

    % 2. MathWorks toolboxes via ver struct (exact actual names, no ver() fallback, no catch swallowing)
    assert(isstruct(all_ver), 'gs3dx:match_export:MissingRuntimeDependency', ...
        'ver_struct must be a struct array.');
    simulink_info = local_find_ver_product(all_ver, 'Simulink');
    simscape_info = local_find_ver_product(all_ver, 'Simscape');
    sm_info = local_find_ver_product(all_ver, 'Simscape Multibody');

    % 3. Python, numpy, and ezc3d versions
    assert(~isempty(py_ver) && strlength(string(py_ver)) > 0, ...
        'gs3dx:match_export:MissingRuntimeDependency', 'Python version could not be determined.');
    assert(~isempty(np_ver) && strlength(string(np_ver)) > 0, ...
        'gs3dx:match_export:MissingRuntimeDependency', 'numpy version could not be determined from loaded cap pipeline.');
    assert(~isempty(ez_ver) && strlength(string(ez_ver)) > 0, ...
        'gs3dx:match_export:MissingRuntimeDependency', 'ezc3d version could not be determined from loaded cap pipeline.');
    py_ver = char(py_ver);
    np_ver = char(np_ver);
    ez_ver = char(ez_ver);

    % 4. Privacy verification (no interpreter paths or license/token secrets)
    local_assert_privacy_clean(v_mat, 'MATLAB version');
    local_assert_privacy_clean(r_mat, 'MATLAB release');
    local_assert_privacy_clean(simulink_info.version, 'Simulink version');
    local_assert_privacy_clean(simscape_info.version, 'Simscape version');
    local_assert_privacy_clean(sm_info.version, 'Simscape Multibody version');
    local_assert_privacy_clean(py_ver, 'Python version');
    local_assert_privacy_clean(np_ver, 'numpy version');
    local_assert_privacy_clean(ez_ver, 'ezc3d version');

    fp = struct();
    fp.matlab_release = char(r_mat);
    fp.matlab_version = char(v_mat);
    fp.matlab = struct('version', char(v_mat), 'release', char(r_mat));
    fp.simulink = simulink_info;
    fp.simscape = simscape_info;
    fp.simscape_multibody = sm_info;
    fp.python_version = char(py_ver);
    fp.numpy_version = char(np_ver);
    fp.ezc3d_version = char(ez_ver);
    fp.scope = 'direct_components_recorded';
    fp.description = 'Direct runtime execution components recorded explicitly; transitive packages are not implied to be frozen.';
end

function prod_info = local_find_ver_product(all_ver, target_name)
    found = false;
    prod_info = struct('name', target_name, 'version', '', 'release', '');
    for i = 1:numel(all_ver)
        if isfield(all_ver(i), 'Name') && strcmp(char(all_ver(i).Name), target_name)
            prod_info.name = char(all_ver(i).Name);
            if isfield(all_ver(i), 'Version') && ~isempty(all_ver(i).Version)
                prod_info.version = char(all_ver(i).Version);
            end
            if isfield(all_ver(i), 'Release') && ~isempty(all_ver(i).Release)
                prod_info.release = char(all_ver(i).Release);
            end
            found = true;
            break;
        end
    end
    assert(found && ~isempty(prod_info.version), 'gs3dx:match_export:MissingRuntimeDependency', ...
        'Required MathWorks product "%s" is not installed or its version could not be determined via ver.', target_name);
end

function local_assert_privacy_clean(val, label)
    s = char(val);
    assert(~contains(s, ':\') && ~contains(s, ':/') && ~contains(s, '/usr/') && ...
           ~contains(s, '/home/') && ~contains(s, '\Users\') && ~contains(s, '.exe'), ...
        'gs3dx:match_export:PrivacyViolation', ...
        'Privacy violation: %s contains absolute interpreter path: "%s"', label, s);
    assert(~contains(lower(s), 'token') && ~contains(lower(s), 'license') && ...
           ~contains(lower(s), 'bearer') && ~contains(lower(s), 'secret'), ...
        'gs3dx:match_export:PrivacyViolation', ...
        'Privacy violation: %s contains sensitive token/license keyword.', label);
end
