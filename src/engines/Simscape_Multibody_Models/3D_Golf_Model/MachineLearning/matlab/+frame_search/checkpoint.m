function checkpoint(runDir, state)
%CHECKPOINT Persist a frame-search checkpoint to <runDir>/checkpoint.mat.
%
%   frame_search.checkpoint(runDir, state) atomically writes the supplied
%   ``state`` struct to ``<runDir>/checkpoint.mat``. The struct must contain
%   at minimum the fields:
%       manifest_sha256       - char/string, hash of the manifest copy
%       last_frame_idx        - non-negative integer, last committed frame
%       previous_torque       - 1xN double row of last committed torques
%       current_state         - opaque struct returned by the state hook
%       committed_torques     - F-by-N double history written so far
%       frame_scores          - F-by-1 double history of best scores
%       wall_clock_per_frame  - F-by-1 double of per-frame wall clock
%
%   The write is atomic: state is first saved to ``checkpoint.mat.tmp`` and
%   then renamed, so a process killed mid-write cannot corrupt the previous
%   checkpoint.

    arguments
        runDir (1, :) char
        state struct
    end

    requiredFields = { ...
        'manifest_sha256', 'last_frame_idx', 'previous_torque', ...
        'current_state', 'committed_torques', 'frame_scores', ...
        'wall_clock_per_frame'};
    for idx = 1:numel(requiredFields)
        if ~isfield(state, requiredFields{idx})
            error('frame_search:checkpoint:MissingField', ...
                'Checkpoint state is missing required field: %s', ...
                requiredFields{idx});
        end
    end
    if state.last_frame_idx < 0
        error('frame_search:checkpoint:NegativeFrame', ...
            'last_frame_idx must be non-negative');
    end

    if ~exist(runDir, 'dir')
        mkdir(runDir);
    end
    finalPath = fullfile(runDir, 'checkpoint.mat');
    tempPath = fullfile(runDir, 'checkpoint.mat.tmp');
    save(tempPath, '-struct', 'state');
    if exist(finalPath, 'file')
        delete(finalPath);
    end
    movefile(tempPath, finalPath);
end
