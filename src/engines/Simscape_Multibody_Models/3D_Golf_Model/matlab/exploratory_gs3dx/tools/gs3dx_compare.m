function cmp = gs3dx_compare(ref, test, opts)
%GS3DX_COMPARE  Signal-by-signal comparison of two flattened runs.
%
%   CMP = GS3DX_COMPARE(REF, TEST) compares two GS3DX_FLATTEN_BUS structs
%   (or two GS3DX_SIMULATE results) on their common signal names:
%     .table     per signal: name, max_abs, rms, range, pass
%     .missing   signals in REF absent from TEST
%     .extra     signals in TEST absent from REF
%     .key_pass  true when every key signal is present and passes
%     .pass      key_pass and every common signal passes
%
%   A signal passes when max_abs <= atol + rtol * range(ref).
%
%   Name-value options:
%     atol (default 1e-6), rtol (default 1e-3)
%     key_signals: leaf-name suffixes that must exist in both runs
%                  (default clubhead and grip-midpoint global position)

    arguments
        ref
        test
        opts.atol (1,1) double {mustBeNonnegative} = 1e-6
        opts.rtol (1,1) double {mustBeNonnegative} = 1e-3
        opts.key_signals (1,:) string = ["CHGlobalPosition", "MPGlobalPosition"]
    end
    ref  = local_flat(ref);
    test = local_flat(test);
    assert(isequal(size(ref.time), size(test.time)) && max(abs(ref.time - test.time)) < 1e-12, ...
        'gs3dx:compare', 'Precondition: runs must share the same time grid');

    [common, ir, it] = intersect(ref.names, test.names, 'stable');
    n = numel(common);
    max_abs = zeros(n, 1); rms_err = zeros(n, 1); rng_ref = zeros(n, 1); pass = false(n, 1);
    for k = 1:n
        a = double(ref.data{ir(k)});
        b = double(test.data{it(k)});
        if ~isequal(size(a), size(b))
            max_abs(k) = Inf; rms_err(k) = Inf; rng_ref(k) = NaN;
            continue;
        end
        e = abs(a - b);
        max_abs(k) = max(e, [], 'all');
        rms_err(k) = sqrt(mean(e .^ 2, 'all'));
        rng_ref(k) = max(a, [], 'all') - min(a, [], 'all');
        pass(k)    = max_abs(k) <= opts.atol + opts.rtol * rng_ref(k);
    end
    cmp = struct();
    cmp.table   = table(common(:), max_abs, rms_err, rng_ref, pass, ...
        'VariableNames', {'name', 'max_abs', 'rms', 'range', 'pass'});
    cmp.missing = setdiff(ref.names, test.names, 'stable');
    cmp.extra   = setdiff(test.names, ref.names, 'stable');

    key_ok = true;
    for s = opts.key_signals
        rows = endsWith(cmp.table.name, "/" + s) | cmp.table.name == s;
        key_ok = key_ok && any(rows) && all(cmp.table.pass(rows));
    end
    cmp.key_pass = key_ok;
    cmp.pass     = key_ok && all(cmp.table.pass);
end

function flat = local_flat(x)
    if isfield(x, 'flat')
        assert(x.status == "success", 'gs3dx:compare', ...
            'Precondition: run %s failed: %s', x.model, x.message);
        flat = x.flat;
    else
        flat = x;
    end
end
