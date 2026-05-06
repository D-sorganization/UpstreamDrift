function [resumeState, resumed] = resume(runDir, manifestSha256, expectedFrameSeconds, staleMultiplier)
%RESUME Try to load a frame-search checkpoint from a run directory.
%
%   [state, resumed] = frame_search.resume(runDir, manifestSha256) attempts
%   to read ``<runDir>/checkpoint.mat``. If the file exists and its
%   ``manifest_sha256`` matches the supplied hash, the loaded state is
%   returned and ``resumed`` is true. Otherwise the function returns an
%   empty struct and ``resumed`` is false (the caller should start fresh).
%
%   [state, resumed] = frame_search.resume(runDir, sha, expectedFrameSeconds,
%   staleMultiplier) additionally inspects ``<runDir>/progress.csv``. If the
%   progress file's mtime is older than ``staleMultiplier *
%   expectedFrameSeconds`` and a checkpoint exists, a warning is emitted but
%   the checkpoint is still returned (the caller restarts from the
%   checkpoint, which is the safe last-known-good frame).

    arguments
        runDir (1, :) char
        manifestSha256 (1, :) char
        expectedFrameSeconds (1, 1) double = 0
        staleMultiplier (1, 1) double = 2.0
    end

    resumeState = struct();
    resumed = false;
    checkpointPath = fullfile(runDir, 'checkpoint.mat');
    if ~isfile(checkpointPath)
        return;
    end

    loaded = load(checkpointPath);
    if ~isfield(loaded, 'manifest_sha256')
        warning('frame_search:resume:NoSha', ...
            'Checkpoint at %s has no manifest_sha256; starting fresh.', ...
            checkpointPath);
        return;
    end
    if ~strcmp(char(loaded.manifest_sha256), manifestSha256)
        warning('frame_search:resume:ShaMismatch', ...
            ['Checkpoint manifest_sha256 (%s) does not match current ', ...
             'manifest (%s); starting fresh.'], ...
            char(loaded.manifest_sha256), manifestSha256);
        return;
    end

    if expectedFrameSeconds > 0
        progressPath = fullfile(runDir, 'progress.csv');
        if isfile(progressPath)
            info = dir(progressPath);
            ageSeconds = (now - info.datenum) * 86400; %#ok<TNOW1>
            if ageSeconds > expectedFrameSeconds * staleMultiplier
                warning('frame_search:resume:StaleLock', ...
                    ['Progress CSV %s has not been updated for %.1fs ', ...
                     '(>%.1fx expected frame time). Restarting from ', ...
                     'last checkpoint.'], progressPath, ageSeconds, ...
                    staleMultiplier);
            end
        end
    end

    resumeState = loaded;
    resumed = true;
end
