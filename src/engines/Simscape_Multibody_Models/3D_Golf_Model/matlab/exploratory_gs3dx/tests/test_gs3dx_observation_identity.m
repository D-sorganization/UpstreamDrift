function tests = test_gs3dx_observation_identity
%TEST_GS3DX_OBSERVATION_IDENTITY  Unit tests for gs3dx_observation_identity and wrapper seam (#10985, #11011, #11161).
%
%   Verifies:
%     1. Omitted default semantics: unmasked capture, EXPORTED_COORDINATE_VALIDITY_ONLY
%     2. Explicit mask binding acceptance: SOURCE_AVAILABILITY_MASK_CALLER_BOUND
%     3. Acceptance of caller-attested raw native-capture source hash without cap.source_sha256
%     4. Fail-closed rejection of C3D export hash mismatch
%     5. Fail-closed rejection of malformed SHA-256 strings (length, hex syntax, non-char)
%     6. Fail-closed rejection of label mismatches and order permutations
%     7. Strengthened label validation: rejects logical, struct, missing, numeric, duplicates, empty strings;
%        accepts nonmissing string vector or cellstr
%     8. Fail-closed rejection of frame count and rate mismatches
%     9. Fail-closed rejection of missing required contract fields
%    10. Fail-closed rejection of empty/typed shape mismatches and invalid sentinels
%    11. Fail-closed rejection of wrong mask types (double vs logical) and invalid 3D dimensions (>3 rejected)
%    12. Single-frame mask acceptance: handles 1 x markers x 1 mask despite MATLAB size dropping trailing singletons
%    13. Deterministic qualification metadata and mask hash generation
%    14. Different mask payloads invalidate mask SHA-256
%    15. Acceptance of different caller source hashes without putting raw source hash into cap
%    16. Known all-true and all-false masks produce distinct hashes and bind rate/clock
%    17. Anonymous channel names accepted when matched exactly
%    18. Wrapper precondition validation for mask and contract matching
%
%   Notes:
%     Pure contract behavior only; does not read private C3D fixtures or run Simscape dynamics.

    tests = functiontests(localfunctions);
end

function setupOnce(t)
    t.TestData.original_path = path;
    here = fileparts(mfilename('fullpath'));
    addpath(here);
end

function teardownOnce(t)
    path(t.TestData.original_path);
end

% -------------------------------------------------------------------------
% Helpers for Synthetic Test Fixtures
% -------------------------------------------------------------------------
function cap = helperSyntheticCap(n_markers, n_frames, rate_hz, labels)
    if nargin < 1 || isempty(n_markers); n_markers = 4; end
    if nargin < 2 || isempty(n_frames); n_frames = 50; end
    if nargin < 3 || isempty(rate_hz); rate_hz = 240.0; end
    if nargin < 4 || isempty(labels)
        labels = ["MarkerA", "MarkerB", "MarkerC", "MarkerD"];
        labels = labels(1:n_markers);
    end

    cap = struct();
    cap.rate_hz = rate_hz;
    cap.n_frames = n_frames;
    cap.labels = labels;
    cap.units = 'm';
    cap.source_units = 'mm';
    cap.observed_mask = [];
    % Note: cap does NOT contain source_sha256; raw native-capture source hash is caller-attested.
end

function c = helperValidContract(cap, export_sha, source_sha)
    if nargin < 2 || isempty(export_sha)
        export_sha = 'abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789';
    end
    if nargin < 3 || isempty(source_sha)
        source_sha = '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
    end
    c = struct();
    c.source_sha256 = source_sha;
    c.export_sha256 = export_sha;
    c.labels = cap.labels;
    c.rate_hz = cap.rate_hz;
    c.n_frames = cap.n_frames;
end

function h = helperExportSha()
    h = 'fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210';
end

% -------------------------------------------------------------------------
% Test 1: Omitted default semantics (unmasked)
% -------------------------------------------------------------------------
function testOmittedDefaultSemantics(t)
    cap = helperSyntheticCap();
    exp_sha = helperExportSha();

    % Call with explicit empty struct contract
    obs = gs3dx_observation_identity(cap, exp_sha, struct([]));

    verifyEqual(t, obs.policy, "EXPORTED_COORDINATE_VALIDITY_ONLY");
    verifyFalse(t, obs.source_mask_applied);
    verifyFalse(t, obs.physical_measurement_verified);
    verifyEqual(t, obs.export_sha256, exp_sha);
    verifyEqual(t, obs.source_sha256, '');
    verifyEqual(t, obs.mask_sha256, '');
    verifyEqual(t, obs.rate_hz, cap.rate_hz);
    verifyEqual(t, obs.n_frames, cap.n_frames);
    verifyEqual(t, obs.n_markers, numel(cap.labels));
    verifyEqual(t, obs.labels, cap.labels);
    verifyFalse(t, isfield(obs, 'file'));
    verifyFalse(t, isfield(obs, 'points'));

    % Call with omitted contract (2 arguments)
    obs2 = gs3dx_observation_identity(cap, exp_sha);
    verifyEqual(t, obs2, obs);

    % Call with omitted export_sha and contract (1 argument)
    obs3 = gs3dx_observation_identity(cap);
    verifyEqual(t, obs3.policy, "EXPORTED_COORDINATE_VALIDITY_ONLY");
    verifyEqual(t, obs3.export_sha256, '');
    verifyFalse(t, obs3.source_mask_applied);
end

% -------------------------------------------------------------------------
% Test 2: Explicit mask binding acceptance (masked)
% -------------------------------------------------------------------------
function testExplicitMaskBindingAcceptance(t)
    cap = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();
    mask = true(1, 4, 50);
    mask(1, 2, 10:20) = false; % introduce specific unobserved samples
    cap.observed_mask = mask;

    caller_source_sha = '1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef';
    contract = helperValidContract(cap, exp_sha, caller_source_sha);

    obs = gs3dx_observation_identity(cap, exp_sha, contract);

    verifyEqual(t, obs.policy, "SOURCE_AVAILABILITY_MASK_CALLER_BOUND");
    verifyTrue(t, obs.source_mask_applied);
    verifyFalse(t, obs.physical_measurement_verified);
    verifyEqual(t, obs.export_sha256, exp_sha);
    verifyEqual(t, obs.source_sha256, caller_source_sha);
    verifyEqual(t, numel(obs.mask_sha256), 64);
    verifyTrue(t, all(ismember(obs.mask_sha256, '0123456789abcdef')));
    verifyEqual(t, obs.rate_hz, cap.rate_hz);
    verifyEqual(t, obs.n_frames, cap.n_frames);
    verifyEqual(t, obs.labels, cap.labels);
    verifyTrue(t, contains(obs.qualification_notice, "SOURCE_AVAILABILITY_MASK_CALLER_BOUND"));
    verifyFalse(t, isfield(obs, 'file'));
    verifyFalse(t, isfield(obs, 'points'));
end

% -------------------------------------------------------------------------
% Test 3: Acceptance of caller-attested raw native-capture source hash without cap.source_sha256
% -------------------------------------------------------------------------
function testCallerBoundSourceHashAccepted(t)
    cap = helperSyntheticCap(4, 50);
    verifyFalse(t, isfield(cap, 'source_sha256'), 'cap must not contain source_sha256 field.');
    exp_sha = helperExportSha();
    cap.observed_mask = true(1, 4, 50);

    caller_sha = 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef';
    contract = helperValidContract(cap, exp_sha, caller_sha);

    obs = gs3dx_observation_identity(cap, exp_sha, contract);

    verifyEqual(t, obs.source_sha256, caller_sha);
    verifyEqual(t, obs.policy, "SOURCE_AVAILABILITY_MASK_CALLER_BOUND");
    verifyTrue(t, contains(obs.qualification_notice, "caller-attested"));
end

% -------------------------------------------------------------------------
% Test 4: Rejection of C3D export hash mismatch
% -------------------------------------------------------------------------
function testRejectionExportMismatch(t)
    cap = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();
    cap.observed_mask = true(1, 4, 50);

    contract = helperValidContract(cap, exp_sha);
    % Contract specifies different C3D export SHA than actual
    contract.export_sha256 = '1111111111111111111111111111111111111111111111111111111111111111';

    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract), ...
        'gs3dx:observation_identity:ExportHashMismatch');
end

% -------------------------------------------------------------------------
% Test 5: Rejection of malformed SHA-256 strings
% -------------------------------------------------------------------------
function testRejectionMalformedHash(t)
    cap = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();
    cap.observed_mask = true(1, 4, 50);

    bad_hashes = { ...
        '0123456789abcdef', ...                       % too short (16 chars)
        ['0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef' '0'], ... % too long (65 chars)
        '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg', ...     % non-hex 'g'
        12345, ...                                    % numeric
        "", ...                                       % empty string
        char(zeros(1, 64)) ...                        % null bytes
    };

    for k = 1:numel(bad_hashes)
        bh = bad_hashes{k};
        % Test malformed source_sha256
        contract1 = helperValidContract(cap, exp_sha);
        contract1.source_sha256 = bh;
        verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract1), ...
            'gs3dx:observation_identity:InvalidHashSyntax');

        % Test malformed export_sha256 in contract
        contract2 = helperValidContract(cap, exp_sha);
        contract2.export_sha256 = bh;
        verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract2), ...
            'gs3dx:observation_identity:InvalidHashSyntax');

        % Test malformed actual export_sha256
        contract3 = helperValidContract(cap, exp_sha);
        verifyError(t, @() gs3dx_observation_identity(cap, bh, contract3), ...
            'gs3dx:observation_identity:InvalidHashSyntax');
    end
end

% -------------------------------------------------------------------------
% Test 6: Rejection of label order and name mismatches
% -------------------------------------------------------------------------
function testRejectionLabelOrderAndNameMismatch(t)
    cap = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();
    cap.observed_mask = true(1, 4, 50);

    % Order permutation: swap MarkerA and MarkerB
    contract_perm = helperValidContract(cap, exp_sha);
    contract_perm.labels = ["MarkerB", "MarkerA", "MarkerC", "MarkerD"];
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_perm), ...
        'gs3dx:observation_identity:LabelMismatch');

    % Renamed label: MarkerD -> MarkerZ
    contract_rename = helperValidContract(cap, exp_sha);
    contract_rename.labels = ["MarkerA", "MarkerB", "MarkerC", "MarkerZ"];
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_rename), ...
        'gs3dx:observation_identity:LabelMismatch');

    % Count mismatch: 3 labels instead of 4
    contract_fewer = helperValidContract(cap, exp_sha);
    contract_fewer.labels = ["MarkerA", "MarkerB", "MarkerC"];
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_fewer), ...
        'gs3dx:observation_identity:LabelMismatch');
end

% -------------------------------------------------------------------------
% Test 7: Strengthened label validation (accept string vector or cellstr; reject logical/struct/missing)
% -------------------------------------------------------------------------
function testStrengthenedLabelRejection(t)
    cap = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();
    cap.observed_mask = true(1, 4, 50);

    % 1. Logical array rejected
    contract_log = helperValidContract(cap, exp_sha);
    contract_log.labels = [true, false, true, false];
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_log), ...
        'gs3dx:observation_identity:InvalidLabels');

    % 2. Struct rejected
    contract_struct = helperValidContract(cap, exp_sha);
    contract_struct.labels = struct('a', 'MarkerA');
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_struct), ...
        'gs3dx:observation_identity:InvalidLabels');

    % 3. Missing values rejected
    contract_missing = helperValidContract(cap, exp_sha);
    contract_missing.labels = ["MarkerA", string(missing), "MarkerC", "MarkerD"];
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_missing), ...
        'gs3dx:observation_identity:InvalidLabels');

    % 4. Numeric array rejected
    contract_num = helperValidContract(cap, exp_sha);
    contract_num.labels = [1, 2, 3, 4];
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_num), ...
        'gs3dx:observation_identity:InvalidLabels');

    % 5. Empty label string rejected
    contract_empty = helperValidContract(cap, exp_sha);
    contract_empty.labels = ["MarkerA", "", "MarkerC", "MarkerD"];
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_empty), ...
        'gs3dx:observation_identity:InvalidLabels');

    % 6. Duplicate labels rejected
    contract_dup = helperValidContract(cap, exp_sha);
    contract_dup.labels = ["MarkerA", "MarkerA", "MarkerC", "MarkerD"];
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_dup), ...
        'gs3dx:observation_identity:InvalidLabels');

    % 7. Cellstr accepted
    contract_cell = helperValidContract(cap, exp_sha);
    contract_cell.labels = {'MarkerA', 'MarkerB', 'MarkerC', 'MarkerD'};
    obs_cell = gs3dx_observation_identity(cap, exp_sha, contract_cell);
    verifyEqual(t, obs_cell.labels, ["MarkerA", "MarkerB", "MarkerC", "MarkerD"]);

    % 8. String vector accepted
    contract_str = helperValidContract(cap, exp_sha);
    contract_str.labels = ["MarkerA", "MarkerB", "MarkerC", "MarkerD"];
    obs_str = gs3dx_observation_identity(cap, exp_sha, contract_str);
    verifyEqual(t, obs_str.labels, contract_str.labels);
end

% -------------------------------------------------------------------------
% Test 8: Rejection of frame count and rate mismatches
% -------------------------------------------------------------------------
function testRejectionFrameAndRateMismatch(t)
    cap = helperSyntheticCap(4, 50, 240.0);
    exp_sha = helperExportSha();
    cap.observed_mask = true(1, 4, 50);

    % Rate mismatch
    contract_rate = helperValidContract(cap, exp_sha);
    contract_rate.rate_hz = 120.0;
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_rate), ...
        'gs3dx:observation_identity:RateMismatch');

    % Frame count mismatch
    contract_frames = helperValidContract(cap, exp_sha);
    contract_frames.n_frames = 60;
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_frames), ...
        'gs3dx:observation_identity:FrameCountMismatch');

    % Negative or nonfinite rate
    contract_bad_rate = helperValidContract(cap, exp_sha);
    contract_bad_rate.rate_hz = -240.0;
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_bad_rate), ...
        'gs3dx:observation_identity:RateMismatch');

    contract_nan_rate = helperValidContract(cap, exp_sha);
    contract_nan_rate.rate_hz = NaN;
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_nan_rate), ...
        'gs3dx:observation_identity:RateMismatch');
end

% -------------------------------------------------------------------------
% Test 9: Rejection of missing required contract fields
% -------------------------------------------------------------------------
function testRejectionMissingContractField(t)
    cap = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();
    cap.observed_mask = true(1, 4, 50);

    fields = {'source_sha256', 'export_sha256', 'labels', 'rate_hz', 'n_frames'};
    for k = 1:numel(fields)
        contract = helperValidContract(cap, exp_sha);
        contract = rmfield(contract, fields{k});
        verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract), ...
            'gs3dx:observation_identity:MissingField');
    end
end

% -------------------------------------------------------------------------
% Test 10: Rejection of empty/typed shape mismatches and invalid sentinels
% -------------------------------------------------------------------------
function testRejectionEmptyAndTypedShapeMismatch(t)
    cap = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();

    % Non-scalar struct contract
    contract = helperValidContract(cap, exp_sha);
    contract_array = [contract, contract];
    cap.observed_mask = true(1, 4, 50);
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, contract_array), ...
        'gs3dx:observation_identity:InvalidContract');

    % Mask without contract (empty struct)
    verifyError(t, @() gs3dx_observation_identity(cap, exp_sha, struct([])), ...
        'gs3dx:observation_identity:MaskWithoutContract');

    % Contract without mask (cap mask is [])
    cap_no_mask = helperSyntheticCap(4, 50);
    verifyError(t, @() gs3dx_observation_identity(cap_no_mask, exp_sha, contract), ...
        'gs3dx:observation_identity:ContractWithoutMask');

    % Contract with logical 0x0 sentinel counts as no-mask -> must reject
    cap_sentinel = helperSyntheticCap(4, 50);
    cap_sentinel.observed_mask = false(0, 0);
    verifyError(t, @() gs3dx_observation_identity(cap_sentinel, exp_sha, contract), ...
        'gs3dx:observation_identity:ContractWithoutMask');

    % Invalid empty sentinels rejected
    bad_sentinels = {false(0, 3), zeros(0, 3), int32([])};
    for k = 1:numel(bad_sentinels)
        cap_bad = helperSyntheticCap(4, 50);
        cap_bad.observed_mask = bad_sentinels{k};
        verifyError(t, @() gs3dx_observation_identity(cap_bad, exp_sha, struct([])), ...
            'gs3dx:observation_identity:BadObservedMask');
    end

    % Logical 0x0 sentinel is accepted as valid empty unmasked
    cap_valid_empty = helperSyntheticCap(4, 50);
    cap_valid_empty.observed_mask = false(0, 0);
    obs = gs3dx_observation_identity(cap_valid_empty, exp_sha, struct([]));
    verifyEqual(t, obs.policy, "EXPORTED_COORDINATE_VALIDITY_ONLY");
    verifyFalse(t, obs.source_mask_applied);
end

% -------------------------------------------------------------------------
% Test 11: Rejection of wrong mask types and invalid dimensions (>3 rejected)
% -------------------------------------------------------------------------
function testRejectionWrongMaskTypeAndDimensions(t)
    cap = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();
    contract = helperValidContract(cap, exp_sha);

    % Double mask instead of logical
    cap_double = cap;
    cap_double.observed_mask = ones(1, 4, 50);
    verifyError(t, @() gs3dx_observation_identity(cap_double, exp_sha, contract), ...
        'gs3dx:observation_identity:InvalidMaskType');

    % 2D mask (4 x 50) instead of 1 x 4 x 50
    cap_2d = cap;
    cap_2d.observed_mask = true(4, 50);
    verifyError(t, @() gs3dx_observation_identity(cap_2d, exp_sha, contract), ...
        'gs3dx:observation_identity:MaskShapeMismatch');

    % Marker dimension mismatch (1 x 5 x 50 instead of 1 x 4 x 50)
    cap_markers = cap;
    cap_markers.observed_mask = true(1, 5, 50);
    verifyError(t, @() gs3dx_observation_identity(cap_markers, exp_sha, contract), ...
        'gs3dx:observation_identity:MaskShapeMismatch');

    % Frame dimension mismatch (1 x 4 x 51 instead of 1 x 4 x 50)
    cap_frames = cap;
    cap_frames.observed_mask = true(1, 4, 51);
    verifyError(t, @() gs3dx_observation_identity(cap_frames, exp_sha, contract), ...
        'gs3dx:observation_identity:MaskShapeMismatch');

    % 4D mask (dimensions > 3 must be rejected)
    cap_4d = cap;
    cap_4d.observed_mask = true(1, 4, 50, 2);
    verifyError(t, @() gs3dx_observation_identity(cap_4d, exp_sha, contract), ...
        'gs3dx:observation_identity:MaskShapeMismatch');
end

% -------------------------------------------------------------------------
% Test 12: Single-frame mask acceptance (1 x markers x 1 handling)
% -------------------------------------------------------------------------
function testSingleFrameMaskAccepted(t)
    % In MATLAB, size(true(1, 4, 1)) returns [1, 4] because trailing singleton
    % dimensions are dropped. The mask validator must accept 1 x markers x 1 via size(x, 1/2/3).
    cap = helperSyntheticCap(4, 1, 100.0);
    exp_sha = helperExportSha();
    mask_1frame = true(1, 4, 1);
    cap.observed_mask = mask_1frame;

    contract = helperValidContract(cap, exp_sha);
    obs = gs3dx_observation_identity(cap, exp_sha, contract);

    verifyEqual(t, obs.policy, "SOURCE_AVAILABILITY_MASK_CALLER_BOUND");
    verifyTrue(t, obs.source_mask_applied);
    verifyEqual(t, obs.n_frames, 1);
    verifyEqual(t, numel(obs.mask_sha256), 64);
    verifyTrue(t, all(ismember(obs.mask_sha256, '0123456789abcdef')));
end

% -------------------------------------------------------------------------
% Test 13: Deterministic identity generation
% -------------------------------------------------------------------------
function testDeterministicIdentity(t)
    cap = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();
    cap.observed_mask = true(1, 4, 50);
    cap.observed_mask(1, 1, 1:10) = false;
    contract = helperValidContract(cap, exp_sha);

    obs1 = gs3dx_observation_identity(cap, exp_sha, contract);
    obs2 = gs3dx_observation_identity(cap, exp_sha, contract);

    verifyEqual(t, obs1, obs2);
    verifyEqual(t, obs1.mask_sha256, obs2.mask_sha256);
end

% -------------------------------------------------------------------------
% Test 14: Different mask payloads invalidate mask SHA-256
% -------------------------------------------------------------------------
function testDifferentMaskChangesIdentity(t)
    cap1 = helperSyntheticCap(4, 50);
    cap2 = helperSyntheticCap(4, 50);
    exp_sha = helperExportSha();

    mask1 = true(1, 4, 50);
    mask2 = true(1, 4, 50);
    mask2(1, 1, 1) = false; % Flip a single boolean sample

    cap1.observed_mask = mask1;
    cap2.observed_mask = mask2;

    contract = helperValidContract(cap1, exp_sha);

    obs1 = gs3dx_observation_identity(cap1, exp_sha, contract);
    obs2 = gs3dx_observation_identity(cap2, exp_sha, contract);

    verifyNotEqual(t, obs1.mask_sha256, obs2.mask_sha256);
end

% -------------------------------------------------------------------------
% Test 15: Different caller source hashes accepted without cap.source_sha256
% -------------------------------------------------------------------------
function testDifferentSourceHashChangesIdentity(t)
    exp_sha = helperExportSha();
    sha_a = '0000000000000000000000000000000000000000000000000000000000000001';
    sha_b = '0000000000000000000000000000000000000000000000000000000000000002';

    % Verify cap does NOT contain source_sha256
    cap = helperSyntheticCap(4, 50, 240.0);
    verifyFalse(t, isfield(cap, 'source_sha256'));
    cap.observed_mask = true(1, 4, 50);

    contract_a = helperValidContract(cap, exp_sha, sha_a);
    contract_b = helperValidContract(cap, exp_sha, sha_b);

    obs_a = gs3dx_observation_identity(cap, exp_sha, contract_a);
    obs_b = gs3dx_observation_identity(cap, exp_sha, contract_b);

    verifyEqual(t, obs_a.source_sha256, sha_a);
    verifyEqual(t, obs_b.source_sha256, sha_b);
    verifyNotEqual(t, obs_a.source_sha256, obs_b.source_sha256);
    % Identical masks produce identical mask SHA
    verifyEqual(t, obs_a.mask_sha256, obs_b.mask_sha256);
end

% -------------------------------------------------------------------------
% Test 16: Known all-true and all-false masks produce distinct hashes
% -------------------------------------------------------------------------
function testKnownAllTrueAndAllFalseMasks(t)
    cap = helperSyntheticCap(4, 10, 240.0);
    exp_sha = helperExportSha();
    contract = helperValidContract(cap, exp_sha);

    cap_true = cap;
    cap_true.observed_mask = true(1, 4, 10);
    obs_true = gs3dx_observation_identity(cap_true, exp_sha, contract);

    cap_false = cap;
    cap_false.observed_mask = false(1, 4, 10);
    obs_false = gs3dx_observation_identity(cap_false, exp_sha, contract);

    verifyNotEqual(t, obs_true.mask_sha256, obs_false.mask_sha256);
    verifyEqual(t, numel(obs_true.mask_sha256), 64);
    verifyEqual(t, numel(obs_false.mask_sha256), 64);

    % Clock and label bindings are preserved
    verifyEqual(t, obs_true.rate_hz, 240.0);
    verifyEqual(t, obs_true.n_frames, 10);
    verifyEqual(t, obs_true.labels, cap.labels);
end

% -------------------------------------------------------------------------
% Test 17: Anonymous channel names accepted when matched exactly
% -------------------------------------------------------------------------
function testAnonymousChannelNamesAccepted(t)
    anon_labels = ["Marker_1:1:", "Channel_02", "*Anonymous*3", "R_Shoulder"];
    cap = helperSyntheticCap(4, 30, 100.0, anon_labels);
    exp_sha = helperExportSha();
    cap.observed_mask = true(1, 4, 30);

    contract = helperValidContract(cap, exp_sha);
    obs = gs3dx_observation_identity(cap, exp_sha, contract);

    verifyEqual(t, obs.policy, "SOURCE_AVAILABILITY_MASK_CALLER_BOUND");
    verifyEqual(t, obs.labels, anon_labels);
end

% -------------------------------------------------------------------------
% Test 18: Wrapper precondition validation for mask and contract matching
% -------------------------------------------------------------------------
function testWrapperPreconditionValidation(t)
    dest = tempname;

    % Mask without contract in match_export
    mask = true(1, 4, 50);
    verifyError(t, @() gs3dx_match_export("capture-A", output_dir=dest, observed_mask=mask), ...
        'gs3dx:match_export:MaskWithoutContract');

    % Contract without mask in match_export
    contract = struct('source_sha256', 'a', 'export_sha256', 'b');
    verifyError(t, @() gs3dx_match_export("capture-A", output_dir=dest, observation_contract=contract), ...
        'gs3dx:match_export:ContractWithoutMask');

    % Bad empty mask sentinel in match_export
    verifyError(t, @() gs3dx_match_export("capture-A", output_dir=dest, observed_mask=false(0, 3)), ...
        'gs3dx:match_export:BadObservedMask');

    % Non-logical mask in match_export
    verifyError(t, @() gs3dx_match_export("capture-A", output_dir=dest, ...
        observed_mask=ones(1, 4, 50), observation_contract=contract), ...
        'gs3dx:match_export:BadObservedMask');

    % Dimensions > 3 in match_export
    verifyError(t, @() gs3dx_match_export("capture-A", output_dir=dest, ...
        observed_mask=true(1, 4, 50, 2), observation_contract=contract), ...
        'gs3dx:match_export:BadObservedMask');

    % Non-scalar contract in match_export
    verifyError(t, @() gs3dx_match_export("capture-A", output_dir=dest, ...
        observed_mask=mask, observation_contract=[contract, contract]), ...
        'gs3dx:match_export:InvalidContract');

    verifyFalse(t, isfolder(dest));
end
