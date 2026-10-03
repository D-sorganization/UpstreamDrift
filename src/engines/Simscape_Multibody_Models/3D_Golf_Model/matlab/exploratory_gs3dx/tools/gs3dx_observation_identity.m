function obs = gs3dx_observation_identity(cap, export_sha256, contract)
%GS3DX_OBSERVATION_IDENTITY  Observation contract verification and qualification metadata (#10985, #11011, #11161).
%
%   OBS = GS3DX_OBSERVATION_IDENTITY(CAP)
%   OBS = GS3DX_OBSERVATION_IDENTITY(CAP, EXPORT_SHA256)
%   OBS = GS3DX_OBSERVATION_IDENTITY(CAP, EXPORT_SHA256, CONTRACT)
%
%   Pure function that validates observation contract bindings against capture
%   metadata and returns whitelisted qualification metadata.
%
%   Qualification Policies:
%     - Without cap.observed_mask and with empty/omitted contract:
%         policy:                        "EXPORTED_COORDINATE_VALIDITY_ONLY"
%         source_mask_applied:           false
%         physical_measurement_verified: false
%     - With explicit cap.observed_mask and matching contract:
%         policy:                        "SOURCE_AVAILABILITY_MASK_CALLER_BOUND"
%         source_mask_applied:           true
%         physical_measurement_verified: false
%
%   Contract Semantics & Attestation Boundaries:
%     - export_sha256 is the actual SHA-256 of the exported C3D capture file
%       as resolved by local_resolve_capture (cap_sha256).
%     - contract.source_sha256 is the caller-attested raw GEARS source hash.
%       It is bound as CALLER_BOUND; this function and the export wrapper do
%       not read, access, or verify the raw GEARS source file on disk.
%     - The observation contract attests decoded upstream availability only.
%       Source hash shape equality does NOT prove actual upstream mask
%       derivation or independent physical measurement accuracy. No false
%       measured or gap-filled guarantees are implied.
%
%   Privacy & Whitelisting:
%     Metadata strictly binds actual C3D export SHA, caller-bound source SHA,
%     mask SHA, and exact ordered labels/rate/count. Private file paths and raw
%     coordinates are never included.
%
%   See also GS3DX_CAPTURE_MARKERS, GS3DX_CAPTURE_POINTS, GS3DX_MATCH_EXPORT.

    arguments
        cap struct
        export_sha256 = ''
        contract struct = struct([])
    end

    % Validate CAP metadata shape and fields fail-closed
    if ~isscalar(cap)
        error('gs3dx:observation_identity:InvalidCapture', ...
            'CAP must be a scalar struct containing capture metadata.');
    end

    % Strict finite positive rate_hz
    if ~isfield(cap, 'rate_hz') || ~isnumeric(cap.rate_hz) || ~isreal(cap.rate_hz) || ...
            ~isscalar(cap.rate_hz) || ~isfinite(cap.rate_hz) || cap.rate_hz <= 0
        error('gs3dx:observation_identity:InvalidRate', ...
            'cap.rate_hz must be a finite positive scalar number.');
    end

    % Strict finite positive integer n_frames
    if ~isfield(cap, 'n_frames') || ~isnumeric(cap.n_frames) || ~isreal(cap.n_frames) || ...
            ~isscalar(cap.n_frames) || ~isfinite(cap.n_frames) || ...
            cap.n_frames < 1 || cap.n_frames ~= fix(cap.n_frames)
        error('gs3dx:observation_identity:InvalidFrameCount', ...
            'cap.n_frames must be a finite positive integer scalar.');
    end

    % Validate marker labels: string vector or cellstr, nonmissing, non-logical, non-struct, unique
    if ~isfield(cap, 'labels')
        error('gs3dx:observation_identity:InvalidLabels', ...
            'cap is missing required field: labels.');
    end
    labels = local_validate_labels(cap.labels, 'cap.labels');

    % Strict units metadata
    units = 'm';
    if isfield(cap, 'units') && (ischar(cap.units) || isstring(cap.units)) && ~isempty(cap.units)
        units = char(cap.units);
    end
    source_units = units;
    if isfield(cap, 'source_units') && (ischar(cap.source_units) || isstring(cap.source_units)) && ~isempty(cap.source_units)
        source_units = char(cap.source_units);
    end

    % Extract and validate cap.observed_mask
    has_mask = false;
    mask = [];
    if isfield(cap, 'observed_mask')
        raw_mask = cap.observed_mask;
        if isempty(raw_mask)
            % Empty sentinel check: only double [] or logical 0x0 permitted
            if ~((islogical(raw_mask) || isa(raw_mask, 'double')) ...
                    && isreal(raw_mask) && isequal(size(raw_mask), [0, 0]))
                error('gs3dx:observation_identity:BadObservedMask', ...
                    'Empty observed_mask must be double [] or logical 0 x 0 sentinel.');
            end
            has_mask = false;
        else
            if ~islogical(raw_mask)
                error('gs3dx:observation_identity:InvalidMaskType', ...
                    'observed_mask must be a logical array.');
            end
            % MATLAB size drops trailing singleton dimensions (e.g. 1 x markers x 1 has size [1, markers]).
            % Mask validation must accept 1 x markers x 1 via size(x, 1/2/3) and reject dimensions > 3.
            if ndims(raw_mask) > 3 || ...
                    size(raw_mask, 1) ~= 1 || ...
                    size(raw_mask, 2) ~= numel(labels) || ...
                    size(raw_mask, 3) ~= cap.n_frames
                error('gs3dx:observation_identity:MaskShapeMismatch', ...
                    'observed_mask size must match expected 1 x %d markers x %d frames (dimensions > 3 rejected).', ...
                    numel(labels), cap.n_frames);
            end
            has_mask = true;
            mask = raw_mask;
        end
    end

    % Mutual exclusivity: explicit mask iff nonempty contract
    has_contract = ~isempty(contract);
    if ~has_mask && has_contract
        error('gs3dx:observation_identity:ContractWithoutMask', ...
            'Nonempty observation contract supplied without an observed mask.');
    end
    if has_mask && ~has_contract
        error('gs3dx:observation_identity:MaskWithoutContract', ...
            'Observed mask supplied without an observation contract.');
    end

    % Branch 1: Unmasked capture (no explicit mask and empty contract)
    if ~has_mask
        actual_export_sha = '';
        if ~isempty(export_sha256)
            if ~local_is_valid_sha256(export_sha256)
                error('gs3dx:observation_identity:InvalidHashSyntax', ...
                    'export_sha256 must be a 64-character hexadecimal string.');
            end
            actual_export_sha = lower(char(export_sha256));
        end

        obs = struct();
        obs.policy = "EXPORTED_COORDINATE_VALIDITY_ONLY";
        obs.source_mask_applied = false;
        obs.physical_measurement_verified = false;
        obs.export_sha256 = actual_export_sha;
        obs.source_sha256 = '';
        obs.mask_sha256 = '';
        obs.rate_hz = double(cap.rate_hz);
        obs.n_frames = double(cap.n_frames);
        obs.n_markers = double(numel(labels));
        obs.labels = labels;
        obs.units = units;
        obs.source_units = source_units;
        obs.qualification_notice = ...
            "EXPORTED_COORDINATE_VALIDITY_ONLY: No source availability mask applied. Export coordinates reflect importer conversion only; no physical measurement verified.";
        return;
    end

    % Branch 2: Masked capture with observation contract
    if ~isstruct(contract) || ~isscalar(contract)
        error('gs3dx:observation_identity:InvalidContract', ...
            'Nonempty observation_contract must be a scalar struct.');
    end

    % Required contract fields
    req_fields = {'source_sha256', 'export_sha256', 'labels', 'rate_hz', 'n_frames'};
    for k = 1:numel(req_fields)
        fn = req_fields{k};
        if ~isfield(contract, fn)
            error('gs3dx:observation_identity:MissingField', ...
                'Contract is missing required field: %s', fn);
        end
    end

    % Validate hash formats (64-char hexadecimal strings)
    if ~local_is_valid_sha256(contract.source_sha256)
        error('gs3dx:observation_identity:InvalidHashSyntax', ...
            'contract.source_sha256 must be a 64-character hexadecimal string.');
    end
    if ~local_is_valid_sha256(contract.export_sha256)
        error('gs3dx:observation_identity:InvalidHashSyntax', ...
            'contract.export_sha256 must be a 64-character hexadecimal string.');
    end
    if isempty(export_sha256) || ~local_is_valid_sha256(export_sha256)
        error('gs3dx:observation_identity:InvalidHashSyntax', ...
            'export_sha256 must be a 64-character hexadecimal string.');
    end

    % Verify actual C3D export hash binding against contract
    if ~strcmpi(char(contract.export_sha256), char(export_sha256))
        error('gs3dx:observation_identity:ExportHashMismatch', ...
            'Contract export_sha256 (%s) does not match actual C3D export SHA-256 (%s).', ...
            char(contract.export_sha256), char(export_sha256));
    end

    % Notice: source_sha256 is caller-attested raw GEARS source hash.
    % It is bound as CALLER_BOUND; no cap.source_sha256 is expected or assigned, and no
    % raw source file read or comparison is performed.

    % Verify sample rate matching
    if ~isnumeric(contract.rate_hz) || ~isreal(contract.rate_hz) || ~isscalar(contract.rate_hz) || ...
            ~isfinite(contract.rate_hz) || contract.rate_hz <= 0 || contract.rate_hz ~= cap.rate_hz
        error('gs3dx:observation_identity:RateMismatch', ...
            'Contract rate_hz (%g) does not match capture rate_hz (%g).', ...
            double(contract.rate_hz), double(cap.rate_hz));
    end

    % Verify frame count matching
    if ~isnumeric(contract.n_frames) || ~isreal(contract.n_frames) || ~isscalar(contract.n_frames) || ...
            ~isfinite(contract.n_frames) || contract.n_frames ~= fix(contract.n_frames) || ...
            contract.n_frames < 1 || contract.n_frames ~= cap.n_frames
        error('gs3dx:observation_identity:FrameCountMismatch', ...
            'Contract n_frames (%d) does not match capture n_frames (%d).', ...
            double(contract.n_frames), double(cap.n_frames));
    end

    % Verify marker labels: strengthened validation and exact native order match
    c_labels = local_validate_labels(contract.labels, 'contract.labels');
    if numel(c_labels) ~= numel(labels) || ~isequal(c_labels, labels)
        error('gs3dx:observation_identity:LabelMismatch', ...
            'Contract labels and native order do not match capture labels.');
    end

    % Compute stable SHA-256 of mask size header plus canonical logical payload bytes
    mask_sha = local_hash_mask(mask);

    obs = struct();
    obs.policy = "SOURCE_AVAILABILITY_MASK_CALLER_BOUND";
    obs.source_mask_applied = true;
    obs.physical_measurement_verified = false;
    obs.export_sha256 = lower(char(contract.export_sha256));
    obs.source_sha256 = lower(char(contract.source_sha256));
    obs.mask_sha256 = mask_sha;
    obs.rate_hz = double(cap.rate_hz);
    obs.n_frames = double(cap.n_frames);
    obs.n_markers = double(numel(labels));
    obs.labels = labels;
    obs.units = units;
    obs.source_units = source_units;
    obs.qualification_notice = ...
        "SOURCE_AVAILABILITY_MASK_CALLER_BOUND: Observation contract attests decoded upstream availability and caller-attested raw GEARS source hash only. No actual raw source read or physical measurement verified.";
end

% -------------------------------------------------------------------------
% Helper: Validate and normalize marker labels fail-closed
% Accepts only nonmissing string vector or cellstr; rejects logical, struct,
% missing, numeric, duplicates, and empty strings.
% -------------------------------------------------------------------------
function labels = local_validate_labels(raw, context_name)
    if nargin < 2
        context_name = 'Marker labels';
    end

    % Strictly reject logical, struct, numeric, or objects
    if islogical(raw) || isstruct(raw) || isnumeric(raw)
        error('gs3dx:observation_identity:InvalidLabels', ...
            '%s cannot be logical, struct, or numeric.', context_name);
    end

    if isstring(raw)
        if ~isvector(raw) || isempty(raw)
            error('gs3dx:observation_identity:InvalidLabels', ...
                '%s must be a non-empty string vector.', context_name);
        end
        if any(ismissing(raw))
            error('gs3dx:observation_identity:InvalidLabels', ...
                '%s must not contain missing values.', context_name);
        end
        labels = reshape(raw, 1, []);
    elseif iscellstr(raw) || iscell(raw)
        if ~isvector(raw) || isempty(raw)
            error('gs3dx:observation_identity:InvalidLabels', ...
                '%s must be a non-empty cellstr vector.', context_name);
        end
        for i = 1:numel(raw)
            elem = raw{i};
            if ~(ischar(elem) && (isrow(elem) || isempty(elem)))
                error('gs3dx:observation_identity:InvalidLabels', ...
                    '%s cell array elements must be character vectors.', context_name);
            end
        end
        labels = reshape(string(raw), 1, []);
    else
        error('gs3dx:observation_identity:InvalidLabels', ...
            '%s must be a string vector or cellstr.', context_name);
    end

    trimmed = strtrim(labels);
    if any(strlength(trimmed) == 0)
        error('gs3dx:observation_identity:InvalidLabels', ...
            '%s must not contain empty or whitespace-only strings.', context_name);
    end

    if numel(unique(labels)) ~= numel(labels)
        error('gs3dx:observation_identity:InvalidLabels', ...
            '%s must contain unique strings.', context_name);
    end
end

% -------------------------------------------------------------------------
% Helper: Validate 64-character hexadecimal SHA-256 string
% -------------------------------------------------------------------------
function ok = local_is_valid_sha256(h)
    ok = false;
    if (ischar(h) && (isrow(h) || isempty(h))) || (isstring(h) && isscalar(h))
        ch = char(h);
        if numel(ch) == 64 && all(ismember(ch, '0123456789abcdefABCDEF'))
            ok = true;
        end
    end
end

% -------------------------------------------------------------------------
% Helper: Compute stable SHA-256 of mask size header and canonical logical bytes
% Size header is '1xNxT:' using size(mask, 1/2/3) to handle trailing singletons.
% -------------------------------------------------------------------------
function hex = local_hash_mask(mask)
    try
        md = java.security.MessageDigest.getInstance('SHA-256');
        size_hdr = uint8(sprintf('%dx%dx%d:', size(mask, 1), size(mask, 2), size(mask, 3)));
        md.update(size_hdr(:));
        payload = uint8(mask(:));
        md.update(payload);
        digest = typecast(md.digest(), 'uint8');
        hex = lower(reshape(dec2hex(digest, 2)', 1, []));
    catch err
        error('gs3dx:observation_identity:HashFailure', ...
            'Failed to compute SHA-256 of observation mask: %s', err.message);
    end
end
