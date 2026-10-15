function export_simscape_closed_loop_snapshot(modelPath, q, v, a, outputPath)
%EXPORT_SIMSCAPE_CLOSED_LOOP_SNAPSHOT Export Simscape Multibody closed-loop snapshot.
%   export_simscape_closed_loop_snapshot(modelPath, q, v, a, outputPath)
%   evaluates M(q), h(q, v), J_c(q), and Jdot(q, v)*v for the closed-loop
%   golf model under MATLAB R2025b (Simscape Multibody) and serializes a
%   canonical JSON snapshot compatible with load_simscape_closed_loop_snapshot.
%
%   Release Gate (#12583 / NSX-55):
%   Strictly requires MATLAB R2025b ('2025b'). Explicitly rejects R2026a or
%   any other release to guarantee deterministic DAE constraint-Jacobian
%   parity with the repository baseline.

    rel = version('-release');
    if ~strcmp(rel, '2025b')
        error('UpstreamDrift:UnsupportedMatlabRelease', ...
            'Simscape closed-loop snapshot export requires MATLAB R2025b (never R2026a); got R%s.', rel);
    end

    if nargin < 5
        error('UpstreamDrift:InvalidArguments', ...
            'Usage: export_simscape_closed_loop_snapshot(modelPath, q, v, a, outputPath)');
    end

    if ~isfile(modelPath)
        error('UpstreamDrift:MissingModel', 'Model file not found: %s', modelPath);
    end

    % Compute SHA-256 digest of the source model file
    md = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(modelPath, 'rb');
    cleaner = onCleanup(@() fclose(fid));
    rawBytes = fread(fid, inf, '*uint8');
    md.update(rawBytes);
    hashBytes = typecast(md.digest(), 'uint8');
    modelSha256 = lower(reshape(dec2hex(hashBytes, 2).', 1, []));

    % Evaluate multibody mass matrix, bias vector, and 6-D weld constraint
    [M, h, J_c, Jdot_v, S, J_g, topology] = evaluate_simscape_multibody_terms(modelPath, q(:), v(:));

    payload = struct();
    payload.schema_version = 'nsx-simscape-snapshot-v1';
    payload.matlab_release = ['R' rel];
    payload.simscape_multibody_version = '25.2';
    payload.model_sha256 = modelSha256;
    payload.inertial_source = 'simscape_multibody_r2025b';
    payload.drift_method = 'analytic';
    payload.topology = topology;
    payload.state = struct('q', q(:).', 'v', v(:).', 'a', a(:).');
    payload.dynamics = struct( ...
        'M', M, ...
        'h', h(:).', ...
        'J_c', J_c, ...
        'Jdot_v', Jdot_v(:).', ...
        'S', S, ...
        'J_g', J_g ...
    );

    jsonText = jsonencode(payload, 'PrettyPrint', true);
    outFid = fopen(outputPath, 'w', 'n', 'UTF-8');
    outCleaner = onCleanup(@() fclose(outFid));
    fwrite(outFid, jsonText, 'char');
end
