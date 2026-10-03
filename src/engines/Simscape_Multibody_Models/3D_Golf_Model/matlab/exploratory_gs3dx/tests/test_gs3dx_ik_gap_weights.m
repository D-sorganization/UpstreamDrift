function tests = test_gs3dx_ik_gap_weights
%TEST_GS3DX_IK_GAP_WEIGHTS Native TDD tests for selective per-target gap position weights.
%   Validates authoritative names, gap weight calculation, robust boundaries,
%   preservation of measured weights, distinct pelvis vs clubhead weights,
%   shuffled names alignment, pre-model load validation on minimal jc, and
%   validation before model loading. Native solver parity is checked separately.
    tests = functiontests(localfunctions);
end


%% Test 1: Authoritative 14 Target Names
function test_authoritative_target_names(t)
    names = gs3dx_ik_target_names();
    verifyTrue(t, iscellstr(names), 'target names must be cell array of char vectors');
    verifyEqual(t, numel(names), 14, 'must have exactly 14 target names');
    verifyEqual(t, numel(names), numel(unique(names)), 'target names must be unique');

    expected = { ...
        'pelvis'; 'hip_L'; 'hip_R'; 'knee_L'; 'knee_R'; ...
        'ankle_L'; 'ankle_R'; 'shoulder_L'; 'shoulder_R'; ...
        'elbow_L'; 'elbow_R'; 'wrist_L'; 'wrist_R'; 'club_head'};
    verifyEqual(t, names(:), expected(:), 'names must match canonical 14 targets in order');
end

%% Test 2: Default Parity and Empty Overrides
function test_default_parity_and_empty_overrides(t)
    names = gs3dx_ik_target_names();

    % Empty struct
    w_empty = gs3dx_ik_gap_weights(names, 0.0, struct());
    verifyEqual(t, size(w_empty), [1, 14]);
    verifyEqual(t, w_empty, zeros(1, 14));

    % Empty array []
    w_nil = gs3dx_ik_gap_weights(names, 0.5, []);
    verifyEqual(t, size(w_nil), [1, 14]);
    verifyEqual(t, w_nil, repmat(0.5, 1, 14));

    % Empty struct array struct([])
    w_empty_arr = gs3dx_ik_gap_weights(names, 0.35, struct([]));
    verifyEqual(t, size(w_empty_arr), [1, 14]);
    verifyEqual(t, w_empty_arr, repmat(0.35, 1, 14));

    % Omitted overrides argument
    w_2arg = gs3dx_ik_gap_weights(names, 0.8);
    verifyEqual(t, w_2arg, repmat(0.8, 1, 14));
end

%% Test 3: Distinct Pelvis vs Clubhead Overrides
function test_distinct_pelvis_vs_club_gap(t)
    names = gs3dx_ik_target_names();
    base_gap = 0.0;
    overrides = struct('pelvis', 1.0, 'club_head', 0.1);

    w = gs3dx_ik_gap_weights(names, base_gap, overrides);
    verifyEqual(t, size(w), [1, 14]);

    idx_pelvis = find(strcmp(names, 'pelvis'));
    idx_club = find(strcmp(names, 'club_head'));
    verifyEqual(t, w(idx_pelvis), 1.0);
    verifyEqual(t, w(idx_club), 0.1);

    % All other 12 targets must retain the base gap weight (0.0)
    other_mask = true(1, 14);
    other_mask([idx_pelvis, idx_club]) = false;
    verifyEqual(t, w(other_mask), zeros(1, 12), ...
        'Non-pelvis and non-clubhead targets must retain base_gap_weight');
end

%% Test 4: Shuffled Names Alignment
function test_shuffled_names_alignment(t)
    names = gs3dx_ik_target_names();
    shuffled_idx = [14, 3, 1, 7, 2, 5, 4, 6, 8, 10, 9, 12, 11, 13];
    names_shuffled = names(shuffled_idx);

    overrides = struct('pelvis', 0.95, 'club_head', 0.15, 'knee_L', 0.45);
    w = gs3dx_ik_gap_weights(names_shuffled, 0.0, overrides);

    for k = 1:numel(names_shuffled)
        nm = names_shuffled{k};
        switch nm
            case 'pelvis'
                verifyEqual(t, w(k), 0.95);
            case 'club_head'
                verifyEqual(t, w(k), 0.15);
            case 'knee_L'
                verifyEqual(t, w(k), 0.45);
            otherwise
                verifyEqual(t, w(k), 0.0);
        end
    end

    % String array input support
    names_str = string(names_shuffled);
    w_str = gs3dx_ik_gap_weights(names_str, 0.0, overrides);
    verifyEqual(t, w_str, w, 'String array names input must produce identical results');
end

%% Test 5: Preservation of Measured Weights
function test_preservation_of_measured_weights(t)
    names = gs3dx_ik_target_names();
    nt = numel(names);
    tw = 1.0 + 0.2 * (1:nt); % Arbitrary distinct target weights

    % Pelvis and club_head are missing/gap, all others are measured
    valid = true(1, nt);
    valid(strcmp(names, 'pelvis')) = false;
    valid(strcmp(names, 'club_head')) = false;

    gap_weights = gs3dx_ik_gap_weights(names, 0.0, struct('pelvis', 1.0, 'club_head', 0.1));

    % Integration formula: weight = (valid + gapvector .* ~valid) .* tw
    weight = (valid + gap_weights .* ~valid) .* tw;

    % Measured targets must preserve their original tw identically
    measured_idx = find(valid);
    verifyEqual(t, weight(measured_idx), tw(measured_idx), ...
        'Measured targets must strictly preserve their exact measured target weights');

    % Gap-filled targets must be scaled by their respective gap weight
    idx_pelvis = find(strcmp(names, 'pelvis'));
    idx_club = find(strcmp(names, 'club_head'));
    verifyEqual(t, weight(idx_pelvis), 1.0 * tw(idx_pelvis));
    verifyEqual(t, weight(idx_club), 0.1 * tw(idx_club));
end

%% Test 6: Boundary Values in [0, 1]
function test_boundary_values(t)
    names = gs3dx_ik_target_names();

    % Exact 0.0 and 1.0 boundaries for base and overrides
    w_zero = gs3dx_ik_gap_weights(names, 0.0, struct('pelvis', 0.0, 'club_head', 1.0));
    verifyEqual(t, w_zero(strcmp(names, 'pelvis')), 0.0);
    verifyEqual(t, w_zero(strcmp(names, 'club_head')), 1.0);

    w_one = gs3dx_ik_gap_weights(names, 1.0, struct('pelvis', 1.0, 'club_head', 0.0));
    verifyEqual(t, w_one(strcmp(names, 'pelvis')), 1.0);
    verifyEqual(t, w_one(strcmp(names, 'club_head')), 0.0);
end

%% Test 7: Robust Input Validation Rejections
function test_validation_errors(t)
    names = gs3dx_ik_target_names();

    % Unknown field: c7
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('c7', 0.5)), 'gs3dx:ik');

    % Unknown field / casing typo: Pelvis
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('Pelvis', 0.5)), 'gs3dx:ik');

    % Negative override value
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('pelvis', -0.01)), 'gs3dx:ik');

    % Value > 1
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('pelvis', 1.01)), 'gs3dx:ik');

    % NaN value
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('pelvis', NaN)), 'gs3dx:ik');

    % Inf value
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('pelvis', Inf)), 'gs3dx:ik');

    % Complex value
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('pelvis', 0.5 + 0.1i)), 'gs3dx:ik');

    % Array value
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('pelvis', [0.1, 0.2])), 'gs3dx:ik');

    % String value
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('pelvis', '0.5')), 'gs3dx:ik');

    % Non-scalar struct
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, struct('pelvis', {0.1, 0.2})), 'gs3dx:ik');

    % Non-struct overrides
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0, [0.1, 0.2]), 'gs3dx:ik');

    % Non-unique names
    verifyError(t, @() gs3dx_ik_gap_weights({'pelvis', 'pelvis'}, 0, []), 'gs3dx:ik');

    % Empty names
    verifyError(t, @() gs3dx_ik_gap_weights({}, 0, []), 'gs3dx:ik');

    % Invalid base_gap_weight
    verifyError(t, @() gs3dx_ik_gap_weights(names, -0.1, []), 'gs3dx:ik');
    verifyError(t, @() gs3dx_ik_gap_weights(names, 1.1, []), 'gs3dx:ik');
    verifyError(t, @() gs3dx_ik_gap_weights(names, NaN, []), 'gs3dx:ik');
    verifyError(t, @() gs3dx_ik_gap_weights(names, Inf, []), 'gs3dx:ik');
    verifyError(t, @() gs3dx_ik_gap_weights(names, 0.5 + 1i, []), 'gs3dx:ik');
    verifyError(t, @() gs3dx_ik_gap_weights(names, [0.1, 0.2], []), 'gs3dx:ik');
end

%% Test 8: Pre-Model Load Validation Fails BEFORE Simulink on Minimal jc
function test_pre_model_load_validation_fails_before_simulink(t)
    % A minimal jc struct with pelvis position points (1 frame)
    jc = struct('pelvis', [0; 0; 0]);

    % Pointing to a dummy model name that does not exist.
    % If validation succeeds, it would attempt load_system and fail with a Simulink error.
    % But invalid gap_target_weight MUST error with 'gs3dx:ik' before any model load.
    non_existent_model = 'NON_EXISTENT_DUMMY_MODEL_NO_LOAD_ALLOWED_XYZ';

    % Unknown field: c7
    verifyError(t, @() gs3dx_whole_body_ik(jc, ...
        'model', non_existent_model, ...
        'gap_target_weight', struct('c7', 0.5)), ...
        'gs3dx:ik', 'Unknown field c7 must error before model load');

    % Negative value on valid field
    verifyError(t, @() gs3dx_whole_body_ik(jc, ...
        'model', non_existent_model, ...
        'gap_target_weight', struct('pelvis', -0.5)), ...
        'gs3dx:ik', 'Negative weight must error before model load');

    % Array value on valid field
    verifyError(t, @() gs3dx_whole_body_ik(jc, ...
        'model', non_existent_model, ...
        'gap_target_weight', struct('pelvis', [0.1, 0.2])), ...
        'gs3dx:ik', 'Non-scalar array must error before model load');
end


function test_reject_empty_identifiers_and_override_types(t)
    verifyError(t,@()gs3dx_ik_gap_weights({'pelvis',''},.1,[]),'gs3dx:ik');
    verifyError(t,@()gs3dx_ik_gap_weights(["pelvis",missing],.1,[]),'gs3dx:ik');
    verifyError(t,@()gs3dx_ik_gap_weights({'pelvis'},.1,{}),'gs3dx:ik');
    verifyError(t,@()gs3dx_ik_gap_weights({'pelvis'},.1,strings(0)),'gs3dx:ik');
end
function test_pelvis_override_retains_club_baseline(t)
    names=gs3dx_ik_target_names();
    w=gs3dx_ik_gap_weights(names,.1,struct('pelvis',1));
    verifyEqual(t,w(strcmp(names,'pelvis')),1);
    verifyEqual(t,w(strcmp(names,'club_head')),.1);
    verifyEqual(t,w(~strcmp(names,'pelvis')),repmat(.1,1,13));
end
