% RUN_ALL  Recreates key results end-to-end
%   This function runs the complete analysis pipeline and saves results
%   to a timestamped output directory.
%
%   Usage:
%       run_all()
%
%   Outputs:
%       Results are saved to output/YYYY-MM-DD/baseline/
%
%   See also: constants

function run_all()
    % Validate inputs (none required, but document function signature)
    arguments
        % No input arguments
    end

    % Load constants (after arguments block)
    % constants.m is in the same directory as run_all.m
    constants;

    % 1) Configure reproducibility
    rng(DEFAULT_RNG_SEED);

    % 2) Prepare output directory
    outdir = fullfile('output', datestr(datetime('now'), 'yyyy-mm-dd'), 'baseline');
    try
        if ~isfolder(outdir)
            mkdir(outdir);
        end
    catch ME
        error('run_all:DirectoryError', 'Failed to create output directory: %s', ME.getReport());
    end

    % 3) Save metadata
    meta.date = datestr(datetime('now'));
    meta.matlab_version = version;
    meta.commit_sha = 'TBD: inject via CI';
    meta.description = 'Baseline run_all template';
    fid = fopen(fullfile(outdir, 'metadata.json'), 'w');
    fprintf(fid, '%s', jsonencode(meta));
    fclose(fid);

    % 4) Placeholder for simulations and plots
    fprintf('run_all completed. Outputs in %s\n', outdir);
end
