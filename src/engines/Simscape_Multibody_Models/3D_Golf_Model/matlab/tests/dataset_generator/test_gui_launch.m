%% Test GUI Launch - Debug Version
% This script will help identify where the GUI is hanging

fprintf('Starting GUI launch test...\n');

try
    fprintf('Step 1: Creating figure...\n');
    fig = figure('Name', 'Test GUI', 'Visible', 'off');

    fprintf('Step 2: Creating colors struct...\n');
    colors = struct();
    colors.primary = [0.2, 0.4, 0.8];
    colors.background = [0.95, 0.95, 0.97];

    fprintf('Step 3: Creating handles struct...\n');
    handles = struct();
    handles.should_stop = false;
    handles.is_paused = false;
    handles.fig = fig;
    handles.colors = colors;
    handles.preferences = struct();
    handles.current_tab = 1;
    handles.checkpoint_data = struct();

    fprintf('Step 4: Loading user preferences...\n');
    % Try to call loadUserPreferences
    try
        handles = loadUserPreferences(handles);
        fprintf('Step 4a: User preferences loaded successfully\n');
    catch ME
        fprintf('Step 4a: Error loading preferences: %s\n', ME.message);
    end

    fprintf('Step 5: Creating main layout...\n');
    % Try to call createMainLayout
    try
        handles = createMainLayout(fig, handles);
        fprintf('Step 5a: Main layout created successfully\n');
    catch ME
        fprintf('Step 5a: Error creating main layout: %s\n', ME.message);
        fprintf('Error details: %s\n', getReport(ME));
    end

    fprintf('Step 6: Storing handles...\n');
    guidata(fig, handles);

    fprintf('Step 7: Applying user preferences...\n');
    try
        applyUserPreferences(handles);
        fprintf('Step 7a: User preferences applied successfully\n');
    catch ME
        fprintf('Step 7a: Error applying preferences: %s\n', ME.message);
    end

    fprintf('Step 8: Updating preview...\n');
    try
        updatePreview([], [], fig);
        fprintf('Step 8a: Preview updated successfully\n');
    catch ME
        fprintf('Step 8a: Error updating preview: %s\n', ME.message);
    end

    fprintf('Step 9: Updating coefficients preview...\n');
    try
        updateCoefficientsPreview([], [], fig);
        fprintf('Step 9a: Coefficients preview updated successfully\n');
    catch ME
        fprintf('Step 9a: Error updating coefficients preview: %s\n', ME.message);
    end

    fprintf('Step 10: Making figure visible...\n');
    set(fig, 'Visible', 'on');

    fprintf('GUI launch test completed successfully!\n');

catch ME
    fprintf('Fatal error in GUI launch test: %s\n', ME.message);
    fprintf('Error details: %s\n', getReport(ME));
end
