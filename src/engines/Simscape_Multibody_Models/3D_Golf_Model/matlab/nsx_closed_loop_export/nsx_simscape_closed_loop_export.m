function out = nsx_simscape_closed_loop_export(specFile, outDir)
%NSX_SIMSCAPE_CLOSED_LOOP_EXPORT  Export a Simscape closed-loop snapshot (NSX-55).
%
%   OUT = NSX_SIMSCAPE_CLOSED_LOOP_EXPORT(SPECFILE, OUTDIR) builds the planar
%   two-arm test rig from SPECFILE, probes it at every state listed in the spec
%   and writes OUTDIR/simscape_closed_loop_export.json in the versioned format
%   "simscape-closed-loop-snapshot/v1" read by
%   src/shared/python/loop_nullspace/simscape_export.py.  OUT is the output path.
%
%   REFUSES TO RUN OUTSIDE MATLAB R2025b (AGENTS.md "Required MATLAB Release"):
%   an export produced by any other release would be rejected as acceptance
%   evidence, so none is produced.
%
%   Provenance recorded: provenance_kind "matlab_export", model_scope
%   "verification_rig" (this rig can never grant golf-model capability), matlab_release,
%   the full version string, the model file name, the SHA-256 of the model
%   file as built in OUTDIR and a UTC timestamp.  The spec JSON carries the
%   topology, the states (q, v, a), the declared bare-club inertia and the rig
%   parameters; M, h, J_c, Jdot_v and the grip origin come from Simscape.
%
%   The golf models (GS3DX_*) cannot be exported this way: they have no
%   actuation or probe ports and are driven by internal Stateflow charts, see
%   docs/development/planning/DV-12583.md.

    arguments
        specFile (1,:) char
        outDir   (1,:) char
    end
    release = ['R' version('-release')];
    if ~strcmp(release, 'R2025b')
        error('nsx_simscape_closed_loop_export:wrongRelease', ...
            ['Required MATLAB release is R2025b (AGENTS.md); running %s. ', ...
             'No export produced.'], release);
    end
    spec = jsondecode(fileread(specFile));
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    modelFile = nsx_build_planar_rig(specFile, outDir);
    [~, name] = fileparts(modelFile);
    load_system(modelFile);
    cleaner = onCleanup(@() close_system(name, 0));

    rows = spec.closure_rows(:).';
    states = cell(1, numel(spec.states));
    for k = 1:numel(spec.states)
        s = spec.states(k);
        dyn = nsx_simscape_probe(name, s.q(:), s.v(:), rows);
        states{k} = struct('q', s.q(:), 'v', s.v(:), 'a', s.a(:), ...
            'M', dyn.M, 'h', dyn.h, 'J_c', dyn.J_c, 'Jdot_v', dyn.Jdot_v(:), ...
            'grip_frame', struct('rotation', eye(3), 'origin', dyn.grip(:)), ...
            'closure_offset_residual', dyn.phi(:), ...
            'jv_residual', dyn.jv_residual);
    end

    doc = struct();
    doc.schema = 'simscape-closed-loop-snapshot/v1';
    doc.provenance = struct( ...
        'provenance_kind', 'matlab_export', ...
        'model_scope', 'verification_rig', ...
        'matlab_release', release, ...
        'matlab_version', version, ...
        'model_file', [name '.slx'], ...
        'model_sha256', file_sha256(modelFile), ...
        'exported_at', char(datetime('now', 'TimeZone', 'UTC', ...
            'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z''')), ...
        'exporter', 'nsx_simscape_closed_loop_export.m', ...
        'note', spec.note);
    doc.topology = spec.topology;
    doc.inertias = spec.inertias;
    doc.states = states;
    if ~isfolder(outDir), mkdir(outDir); end
    out = fullfile(outDir, 'simscape_closed_loop_export.json');
    fid = fopen(out, 'w');
    assert(fid > 0, 'nsx_simscape_closed_loop_export:io', 'cannot write %s', out);
    fwrite(fid, jsonencode(doc, 'PrettyPrint', true));
    fclose(fid);
end

function hex = file_sha256(path)
    fid = fopen(path, 'r');
    assert(fid > 0, 'nsx_simscape_closed_loop_export:io', 'cannot read %s', path);
    bytes = fread(fid, Inf, '*uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    digest = typecast(md.digest(), 'uint8');
    hex = lower(sprintf('%02x', digest));
end
