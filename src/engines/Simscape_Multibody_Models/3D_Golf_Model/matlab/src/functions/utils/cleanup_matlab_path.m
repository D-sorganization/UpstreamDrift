%% Cleanup MATLAB Path - Remove Backup Scripts
% This script removes all backup-related paths from the MATLAB path
% to prevent conflicts and improve performance

fprintf('Cleaning up MATLAB path...\n');

% Get current path
current_path = path;
paths = strsplit(current_path, ';');

% Find backup paths
backup_paths = paths(contains(paths, 'Backup'));

fprintf('Found %d backup paths to remove:\n', length(backup_paths));

% Remove each backup path
for i = 1:length(backup_paths)
    if contains(backup_paths{i}, 'Backup')
        fprintf('Removing: %s\n', backup_paths{i});
        rmpath(backup_paths{i});
    end
end

% Verify cleanup
new_path = path;
new_paths = strsplit(new_path, ';');
remaining_backup = new_paths(contains(new_paths, 'Backup'));

fprintf('\nCleanup complete!\n');
fprintf('Remaining backup paths: %d\n', length(remaining_backup));

if length(remaining_backup) == 0
    fprintf('✓ All backup paths successfully removed!\n');
else
    fprintf('⚠ Some backup paths remain:\n');
    for i = 1:length(remaining_backup)
        fprintf('  %s\n', remaining_backup{i});
    end
end

fprintf('\nCurrent MATLAB path length: %d entries\n', length(new_paths));
