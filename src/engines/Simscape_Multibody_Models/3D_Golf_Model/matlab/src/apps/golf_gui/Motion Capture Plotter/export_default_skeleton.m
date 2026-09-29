function export_default_skeleton(pose_name, out_path)
%EXPORT_DEFAULT_SKELETON  Run a 5 ms sim and dump joint positions to JSON.
%
%   EXPORT_DEFAULT_SKELETON()              -> writes Impact skeleton
%   EXPORT_DEFAULT_SKELETON('Impact')      -> writes Impact skeleton
%   EXPORT_DEFAULT_SKELETON('TopofBackswing')
%                                          -> writes TopofBackswing skeleton
%   EXPORT_DEFAULT_SKELETON(POSE, OUT_PATH) -> custom output path
%
%   POSE_NAME selects which model-input MAT to load.  The corresponding
%   files are expected at:
%       matlab/src/model/inputs/3DModelInputs_<POSE>.mat
%
%   The output JSON contains joint world-frame positions (metres) at t=0:
%       hip, spine, hub, ls/rs (shoulders), le/re (elbows),
%       lw/rw (wrists), mp (mid-hands), butt, ch (clubhead)
%
%   The starting_pose_matcher.py tool reads the JSON.  Run this script
%   once per pose; the JSON files are committed if you want them in CI.

    if nargin < 1 || isempty(pose_name)
        pose_name = 'Impact';
    end
    pose_name = char(pose_name);

    here = fileparts(mfilename('fullpath'));
    if nargin < 2 || isempty(out_path)
        out_path = fullfile(here, sprintf('simscape_skeleton_%s.json', pose_name));
    end

    fprintf('=== export_default_skeleton(%s) ===\n', pose_name);
    fprintf('Loading starting position (this runs a tiny 5ms sim)...\n');

    % Path setup -------------------------------------------------------------
    matlab_root = fileparts(fileparts(fileparts(fileparts(here))));   % .../matlab/
    src_dir = fullfile(matlab_root, 'src');
    if exist(src_dir, 'dir');  addpath(genpath(src_dir));  end
    mm_shared = fullfile(matlab_root, 'motion_matching', 'shared');
    if exist(mm_shared, 'dir'); addpath(mm_shared); end

    % Resolve input MAT file -------------------------------------------------
    input_file = fullfile(matlab_root, 'src', 'model', 'inputs', ...
                          sprintf('3DModelInputs_%s.mat', pose_name));
    if ~isfile(input_file)
        error('export_default_skeleton:noInputMat', ...
              'Input MAT not found: %s', input_file);
    end
    fprintf('input_file: %s\n', input_file);

    % Run the sim -----------------------------------------------------------
    skel = load_impact_starting_position(struct( ...
        'input_file', input_file, ...
        'verbose', true));

    % Build connectivity ----------------------------------------------------
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
    out.pose       = string(pose_name);
    out.joints     = struct();
    joint_fields = {'hip','spine','hub','ls','rs','le','re', ...
                    'lw','rw','mp','ch','butt'};
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

    % Write JSON -----------------------------------------------------------
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
