function export_default_skeleton(out_path)
%EXPORT_DEFAULT_SKELETON  Run a 5 ms sim and dump default joint positions to JSON.
%
%   EXPORT_DEFAULT_SKELETON() saves to simscape_default_skeleton.json next
%   to this script.
%   EXPORT_DEFAULT_SKELETON(OUT_PATH) saves to the specified path.
%
%   The output JSON has joint world-frame positions (metres) at t=0 of the
%   GolfSwing3D_Kinetic model with the standard Impact MAT inputs:
%
%       {
%         "joints": {"hip":[x,y,z], "spine":[x,y,z], "hub":[x,y,z],
%                    "ls":[x,y,z], "rs":[x,y,z], "le":[x,y,z], "re":[x,y,z],
%                    "lw":[x,y,z], "rw":[x,y,z], "mp":[x,y,z], "ch":[x,y,z]},
%         "segments": [["hip","spine"], ["spine","hub"], ...],
%         "model_name": "GolfSwing3D_Kinetic",
%         "input_file": "...",
%         "exported_at": "2026-05-07T..."
%       }
%
%   Run this ONCE; the starting_pose_matcher.py reads the JSON.

    if nargin < 1
        here = fileparts(mfilename('fullpath'));
        out_path = fullfile(here, 'simscape_default_skeleton.json');
    end

    fprintf('=== export_default_skeleton ===\n');
    fprintf('Loading default starting position (this runs a tiny 5ms sim)...\n');

    % Make sure path is set up
    here = fileparts(mfilename('fullpath'));
    matlab_root = fileparts(fileparts(fileparts(fileparts(here))));   % .../matlab/
    src_dir = fullfile(matlab_root, 'src');
    if exist(src_dir, 'dir')
        addpath(genpath(src_dir));
    end
    mm_shared = fullfile(matlab_root, 'motion_matching', 'shared');
    if exist(mm_shared, 'dir')
        addpath(mm_shared);
    end

    skel = load_impact_starting_position(struct('verbose', true));

    % Build the connectivity list for plotting (parent -> child segments).
    segments = { ...
        {'hip',   'spine'}, ...
        {'spine', 'hub'  }, ...
        {'hub',   'ls'   }, ...
        {'hub',   'rs'   }, ...
        {'ls',    'le'   }, ...
        {'rs',    're'   }, ...
        {'le',    'lw'   }, ...
        {'re',    'rw'   }, ...
        {'lw',    'mp'   }, ...
        {'rw',    'mp'   }, ...
        {'mp',    'ch'   } };

    out = struct();
    out.joints     = struct();
    joint_fields = {'hip','spine','hub','ls','rs','le','re','lw','rw','mp','ch','butt'};
    for k = 1:numel(joint_fields)
        f = joint_fields{k};
        if isfield(skel, f) && numel(skel.(f)) == 3
            out.joints.(f) = double(skel.(f)(:)');
        end
    end
    out.segments    = segments;
    out.model_name  = char(skel.model_name);
    out.input_file  = char(skel.input_file);
    out.exported_at = char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ss'));

    % Write JSON.
    txt = jsonencode(out, 'PrettyPrint', true);
    fid = fopen(out_path, 'w');
    if fid < 0
        error('export_default_skeleton:cannotWrite', ...
              'Could not open %s for writing.', out_path);
    end
    fprintf(fid, '%s\n', txt);
    fclose(fid);

    fprintf('Wrote: %s\n', out_path);
    fprintf('Joints: %s\n', strjoin(fieldnames(out.joints), ', '));
end
