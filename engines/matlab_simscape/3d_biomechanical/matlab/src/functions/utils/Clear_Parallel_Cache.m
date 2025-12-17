% 1. Clear everything in the main MATLAB session
clear all
clear functions
clear classes

% 2. Restart the parallel pool to force workers to reload
if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'));
end

% 3. Start fresh parallel pool
parpool('local', 4);  % or however many workers you want

% 4. Now run your GUI
Dataset_GUI
