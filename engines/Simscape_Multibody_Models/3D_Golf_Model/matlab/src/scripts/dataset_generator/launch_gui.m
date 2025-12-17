%% Launch Enhanced Golf Swing Data Generator GUI
% This script launches the enhanced version of the GUI with tabbed interface
% and advanced post-processing capabilities

% Add current directory to path if not already there
current_dir = fileparts(mfilename('fullpath'));
if ~contains(path, current_dir)
    addpath(current_dir);
end

% Launch the enhanced GUI
try
    Dataset_GUI();
    fprintf('Enhanced Golf Swing Data Generator launched successfully!\n');
    fprintf('Features available:\n');
    fprintf('  - Tabbed interface (Data Generation / Post-Processing)\n');
    fprintf('  - Pause/Resume functionality with checkpoints\n');
    fprintf('  - Multiple export formats (CSV, Parquet, MAT, JSON)\n');
    fprintf('  - Batch processing with configurable batch sizes\n');
    fprintf('  - Feature extraction for machine learning\n');
    fprintf('  - Memory-efficient processing\n');
catch ME
    fprintf('Error launching enhanced GUI: %s\n', ME.message);
    fprintf('Please check the error and try again.\n');
end
