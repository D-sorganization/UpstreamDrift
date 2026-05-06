function files = scan_results_directory(results_dir)
%SCAN_RESULTS_DIRECTORY  Discover result .mat files for the leaderboard.
%
%   FILES = SCAN_RESULTS_DIRECTORY(RESULTS_DIR) returns a string column
%   vector of absolute paths to every *.mat file found beneath
%   RESULTS_DIR (recursively). The returned list is sorted lexicographically
%   so leaderboard ordering is deterministic before any sort_by is applied.
%
%   When RESULTS_DIR does not exist or contains no .mat files the function
%   returns an empty (0x1) string array — callers handle "no results" by
%   short-circuiting to an empty schema-correct table.
%
%   Preconditions:
%     - RESULTS_DIR is a 1x1 string. Existence is *not* required: an empty
%       result is preferred over an error so the leaderboard renders
%       gracefully on a fresh checkout.
%
%   Postconditions:
%     - Output is a string column vector (Nx1, possibly 0x1).
%     - Each element exists on disk and ends with ".mat".
    arguments
        results_dir (1,1) string
    end

    files = strings(0, 1);
    if ~isfolder(results_dir)
        return;
    end

    listing = dir(fullfile(char(results_dir), '**', '*.mat'));
    if isempty(listing)
        return;
    end

    paths = strings(numel(listing), 1);
    for k = 1:numel(listing)
        paths(k) = string(fullfile(listing(k).folder, listing(k).name));
    end
    files = sort(paths);

    % Postcondition: every entry exists and is a .mat file.
    for k = 1:numel(files)
        assert(isfile(files(k)), ...
            "scan_results_directory:postcondition", ...
            "Discovered file does not exist: %s", files(k));
        assert(endsWith(files(k), ".mat", 'IgnoreCase', true), ...
            "scan_results_directory:postcondition", ...
            "Non-.mat path returned: %s", files(k));
    end
end
