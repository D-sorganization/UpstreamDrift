function tests = test_gs3dx_fit_human_visuals
% Unit and integration tests for gs3dx_fit_human_visuals and gs3dx_scale_vector.
    tests = functiontests(localfunctions);
end

function setupOnce(t)
    t.TestData.original_path = path;
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'tools'));
end

function teardownOnce(t)
    path(t.TestData.original_path);
end

% -------------------------------------------------------------------------
% 1. Pure Vector Scaling Unit Tests (Pure Numerics, No Simulink Engine Required)
% -------------------------------------------------------------------------

function testVectorScalePreservesTransverseAxes(t)
    % Scaling along axis 3 must strictly scale element 3 and leave 1 and 2 untouched
    v = [0.048, 0.048, 0.165];
    scaled = gs3dx_scale_vector(v, 1.15, 3);

    verifyEqual(t, scaled(1:2), [0.048, 0.048], 'AbsTol', 1e-15);
    verifyEqual(t, scaled(3), 0.165 * 1.15, 'AbsTol', 1e-15);
end

function testVectorScaleIdentity(t)
    % Scale factor 1.0 must return the identical vector
    v = [0.17, 0.115, 0.12];
    scaled = gs3dx_scale_vector(v, 1.0, 3);

    verifyEqual(t, scaled, v, 'AbsTol', 1e-15);
end

function testVectorScaleAxis1And2(t)
    % Scaling along axis 1 or 2 must scale only that component
    v = [1.0, 2.0, 3.0];
    s1 = gs3dx_scale_vector(v, 2.0, 1);
    verifyEqual(t, s1, [2.0, 2.0, 3.0], 'AbsTol', 1e-15);

    s2 = gs3dx_scale_vector(v, 0.5, 2);
    verifyEqual(t, s2, [1.0, 1.0, 3.0], 'AbsTol', 1e-15);
end

function testVectorScaleInvalidVector(t)
    % Input vector must be 1x3 real finite numeric
    bad_vectors = { ...
        [1, 2], ...             % 1x2
        [1, 2, 3, 4], ...       % 1x4
        zeros(2, 3), ...        % 2x3
        [], ...                 % empty
        [1, NaN, 3], ...        % NaN
        [1, 2, Inf], ...        % Inf
        [1, 2+1i, 3], ...       % complex
        "not_numeric" ...       % string
    };

    for i = 1:numel(bad_vectors)
        verifyError(t, @() gs3dx_scale_vector(bad_vectors{i}, 1.1, 3), ...
            'gs3dx:scale_vector:InvalidVector');
    end
end

function testVectorScaleInvalidScaleFactor(t)
    % Scale factor must be positive finite real numeric scalar
    bad_factors = {-1.0, 0, NaN, Inf, 1+2i, [1.1, 1.2], "bad"};

    for i = 1:numel(bad_factors)
        verifyError(t, @() gs3dx_scale_vector([1, 2, 3], bad_factors{i}, 3), ...
            'gs3dx:scale_vector:InvalidScaleFactor');
    end
end

function testVectorScaleInvalidAxis(t)
    % Axis index must be 1, 2, or 3
    bad_axes = {0, 4, -1, 1.5, NaN, Inf, [1, 2], "3"};

    for i = 1:numel(bad_axes)
        verifyError(t, @() gs3dx_scale_vector([1, 2, 3], 1.1, bad_axes{i}), ...
            'gs3dx:scale_vector:InvalidAxis');
    end
end

% -------------------------------------------------------------------------
% 2. Adapter Input Precondition Tests (Fail-Closed Offline Checks)
% -------------------------------------------------------------------------

function testFailClosedOnUnloadedModel(t)
    fit_mock = struct('vars', struct( ...
        'FitHubtoSLength', 6.3, 'FitUpperArmLength', 12.0, 'FitLowerArmLength', 11.0, ...
        'FitLowerTorsoLength', 9.6, 'FitUpperTorsoLength', 9.6, ...
        'ThighLength', 0.46, 'ShankLength', 0.42));

    verifyError(t, @() gs3dx_fit_human_visuals('NonExistentModel', fit_mock), ...
        'gs3dx:fit_human_visuals:ModelNotLoaded');
    verifyError(t, @() gs3dx_fit_human_visuals('', fit_mock), ...
        'gs3dx:fit_human_visuals:ModelNotLoaded');
end

function testFailClosedOnMissingField(t)
    fit_incomplete = struct('vars', struct( ...
        'FitHubtoSLength', 6.3, 'FitUpperArmLength', 12.0));

    verifyError(t, @() gs3dx_fit_human_visuals('GS3DX_Human', fit_incomplete), ...
        'gs3dx:fit_human_visuals:MissingField');
end

function testFailClosedOnInvalidLengths(t)
    bad_vars = { ...
        struct('FitHubtoSLength', -6.3, 'FitUpperArmLength', 12.0, 'FitLowerArmLength', 11.0, ...
            'FitLowerTorsoLength', 9.6, 'FitUpperTorsoLength', 9.6, 'ThighLength', 0.46, 'ShankLength', 0.42), ...
        struct('FitHubtoSLength', 6.3, 'FitUpperArmLength', 0, 'FitLowerArmLength', 11.0, ...
            'FitLowerTorsoLength', 9.6, 'FitUpperTorsoLength', 9.6, 'ThighLength', 0.46, 'ShankLength', 0.42), ...
        struct('FitHubtoSLength', 6.3, 'FitUpperArmLength', NaN, 'FitLowerArmLength', 11.0, ...
            'FitLowerTorsoLength', 9.6, 'FitUpperTorsoLength', 9.6, 'ThighLength', 0.46, 'ShankLength', 0.42), ...
        struct('FitHubtoSLength', 6.3, 'FitUpperArmLength', Inf, 'FitLowerArmLength', 11.0, ...
            'FitLowerTorsoLength', 9.6, 'FitUpperTorsoLength', 9.6, 'ThighLength', 0.46, 'ShankLength', 0.42), ...
        struct('FitHubtoSLength', 6.3, 'FitUpperArmLength', 12+1i, 'FitLowerArmLength', 11.0, ...
            'FitLowerTorsoLength', 9.6, 'FitUpperTorsoLength', 9.6, 'ThighLength', 0.46, 'ShankLength', 0.42), ...
        struct('FitHubtoSLength', 6.3, 'FitUpperArmLength', [12, 13], 'FitLowerArmLength', 11.0, ...
            'FitLowerTorsoLength', 9.6, 'FitUpperTorsoLength', 9.6, 'ThighLength', 0.46, 'ShankLength', 0.42)};

    for i = 1:numel(bad_vars)
        fit_bad = struct('vars', bad_vars{i});
        verifyError(t, @() gs3dx_fit_human_visuals('GS3DX_Human', fit_bad), ...
            'gs3dx:fit_human_visuals:InvalidLengths');
    end
end

% -------------------------------------------------------------------------
% 3. Native Model Adapter Integration Test (Capability-Sensitive, Parent Runs)
% -------------------------------------------------------------------------

function testNativeHumanVisualAdaptationPreservesInvariants(t)
    mdl = 'GS3DX_Human';
    mdl_file = which([mdl '.slx']);
    assumeTrue(t, ~isempty(mdl_file) && isfile(mdl_file), ...
        'Native model GS3DX_Human.slx must be on the MATLAB path to execute native integration test');

    if bdIsLoaded(mdl)
        close_system(mdl, 0);
    end
    load_system(mdl);
    t.addTeardown(@() close_system(mdl, 0));

    % Read baseline parameters before adaptation
    base_r_arm = slResolve(get_param([mdl '/L Upper Arm'], 'EllipsoidRadii'), [mdl '/L Upper Arm']);
    base_r_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'EllipsoidRadii'), [mdl '/Lower Body/L Thigh']);
    base_m_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'Mass'), [mdl '/Lower Body/L Thigh']);
    base_com_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'CenterOfMass'), [mdl '/Lower Body/L Thigh']);
    base_moi_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'MomentsOfInertia'), [mdl '/Lower Body/L Thigh']);
    base_poi_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'ProductsOfInertia'), [mdl '/Lower Body/L Thigh']);
    base_joint_xform = get_param([mdl '/Lower Body/L Knee Mount'], 'TranslationCartesianOffset');

    % Prepare modified fitted lengths (+10% upper arm, +8% thigh)
    ws = get_param(mdl, 'ModelWorkspace');
    base_ua = ws.getVariable('FitUpperArmLength');
    if isa(base_ua, 'Simulink.Parameter'), base_ua = base_ua.Value; end
    base_tl = ws.getVariable('ThighLength');
    if isa(base_tl, 'Simulink.Parameter'), base_tl = base_tl.Value; end
    base_sl = ws.getVariable('ShankLength');
    if isa(base_sl, 'Simulink.Parameter'), base_sl = base_sl.Value; end
    base_hub = ws.getVariable('FitHubtoSLength');
    if isa(base_hub, 'Simulink.Parameter'), base_hub = base_hub.Value; end
    base_la = ws.getVariable('FitLowerArmLength');
    if isa(base_la, 'Simulink.Parameter'), base_la = base_la.Value; end
    base_lt = ws.getVariable('FitLowerTorsoLength');
    if isa(base_lt, 'Simulink.Parameter'), base_lt = base_lt.Value; end
    base_ut = ws.getVariable('FitUpperTorsoLength');
    if isa(base_ut, 'Simulink.Parameter'), base_ut = base_ut.Value; end

    fit_test = struct('vars', struct( ...
        'FitHubtoSLength', double(base_hub), ...
        'FitUpperArmLength', double(base_ua) * 1.10, ...
        'FitLowerArmLength', double(base_la), ...
        'FitLowerTorsoLength', double(base_lt), ...
        'FitUpperTorsoLength', double(base_ut), ...
        'ThighLength', double(base_tl) * 1.08, ...
        'ShankLength', double(base_sl)));

    % Execute visual adaptation
    report = gs3dx_fit_human_visuals(mdl, fit_test);
    verifyEqual(t, report.status, "adapted");

    % 1. Verify visual radii actually scaled longitudinally
    new_r_arm = slResolve(get_param([mdl '/L Upper Arm'], 'EllipsoidRadii'), [mdl '/L Upper Arm']);
    verifyEqual(t, new_r_arm(1:2), base_r_arm(1:2), 'AbsTol', 1e-12, 'Upper arm transverse radii must be invariant');
    verifyEqual(t, new_r_arm(3), base_r_arm(3) * 1.10, 'AbsTol', 1e-6, 'Upper arm longitudinal radius must scale by 1.10');

    new_r_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'EllipsoidRadii'), [mdl '/Lower Body/L Thigh']);
    verifyEqual(t, new_r_thigh(1:2), base_r_thigh(1:2), 'AbsTol', 1e-12, 'Thigh transverse radii must be invariant');
    verifyEqual(t, new_r_thigh(3), base_r_thigh(3) * 1.08, 'AbsTol', 1e-6, 'Thigh longitudinal radius must scale by 1.08');

    % 2. Verify mass, COM, moments, products of inertia are 100% IDENTICALLY PRESERVED
    new_m_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'Mass'), [mdl '/Lower Body/L Thigh']);
    new_com_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'CenterOfMass'), [mdl '/Lower Body/L Thigh']);
    new_moi_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'MomentsOfInertia'), [mdl '/Lower Body/L Thigh']);
    new_poi_thigh = slResolve(get_param([mdl '/Lower Body/L Thigh'], 'ProductsOfInertia'), [mdl '/Lower Body/L Thigh']);

    verifyEqual(t, new_m_thigh, base_m_thigh, 'AbsTol', 1e-12, 'Thigh mass must remain strictly preserved');
    verifyEqual(t, new_com_thigh, base_com_thigh, 'AbsTol', 1e-12, 'Thigh center of mass must remain strictly preserved');
    verifyEqual(t, new_moi_thigh, base_moi_thigh, 'AbsTol', 1e-12, 'Thigh moments of inertia must remain strictly preserved');
    verifyEqual(t, new_poi_thigh, base_poi_thigh, 'AbsTol', 1e-12, 'Thigh products of inertia must remain strictly preserved');

    % 3. Verify physical joint frame transforms are 100% UNTOUCHED
    new_joint_xform = get_param([mdl '/Lower Body/L Knee Mount'], 'TranslationCartesianOffset');
    verifyEqual(t, new_joint_xform, base_joint_xform, 'Physical joint transform must remain completely untouched');

    % 4. Verify workspace received fitted variables
    verifyEqual(t, ws.getVariable('FitUpperArmLength'), double(base_ua) * 1.10, 'AbsTol', 1e-12);
    verifyEqual(t, ws.getVariable('ThighLength'), double(base_tl) * 1.08, 'AbsTol', 1e-12);
end
