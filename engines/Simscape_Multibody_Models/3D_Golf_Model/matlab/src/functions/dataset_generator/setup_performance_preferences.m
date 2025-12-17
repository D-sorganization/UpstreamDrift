function setup_performance_preferences()
    % SETUP_PERFORMANCE_PREFERENCES - Quick setup for performance preferences
    %
    % This script allows you to easily configure performance preferences
    % for the Golf Swing Data Generator GUI.
    %
    % Usage:
    %   setup_performance_preferences()

    fprintf('Performance Preferences Setup\n');
    fprintf('============================\n\n');

    % Get the script directory
    script_dir = fileparts(mfilename('fullpath'));
    pref_file = fullfile(script_dir, 'user_preferences.mat');

    % Load existing preferences or create defaults
    if exist(pref_file, 'file')
        fprintf('Loading existing preferences...\n');
        loaded_prefs = load(pref_file);
        preferences = loaded_prefs.preferences;
    else
        fprintf('Creating new preferences file...\n');
        preferences = struct();
    end

    % Set up default performance preferences
    fprintf('\nSetting up performance preferences:\n');

    % Preallocation settings
    preferences.enable_preallocation = true;
    preferences.preallocation_buffer_size = 1000;
    fprintf('✓ Preallocation enabled (buffer size: %d)\n', preferences.preallocation_buffer_size);

    % Caching settings
    preferences.enable_model_caching = true;
    fprintf('✓ Model caching enabled\n');

    % Parallel processing settings
    preferences.enable_parallel_processing = true;
    preferences.max_parallel_workers = feature('numcores'); % Use all available cores
    preferences.cluster_profile = 'Local_Cluster'; % Use your local cluster profile
    preferences.use_local_cluster = true;
    fprintf('✓ Parallel processing enabled (%d workers using Local_Cluster)\n', preferences.max_parallel_workers);

    % Compression settings
    preferences.enable_data_compression = true;
    preferences.compression_level = 6;
    fprintf('✓ Data compression enabled (level: %d)\n', preferences.compression_level);

    % Memory management settings
    preferences.enable_memory_pooling = true;
    preferences.memory_pool_size = 100; % MB
    fprintf('✓ Memory pooling enabled (%d MB)\n', preferences.memory_pool_size);

    % Performance monitoring settings
    preferences.enable_performance_monitoring = true;
    preferences.enable_memory_monitoring = true;
    fprintf('✓ Performance monitoring enabled\n');

    % Save preferences
    save(pref_file, 'preferences');
    fprintf('\n✓ Preferences saved to: %s\n', pref_file);

    % Display current settings
    fprintf('\nCurrent Performance Settings:\n');
    fprintf('=============================\n');
    fprintf('Preallocation: %s\n', yesno(preferences.enable_preallocation));
    fprintf('Model Caching: %s\n', yesno(preferences.enable_model_caching));
    fprintf('Parallel Processing: %s\n', yesno(preferences.enable_parallel_processing));
    fprintf('Data Compression: %s\n', yesno(preferences.enable_data_compression));
    fprintf('Memory Pooling: %s\n', yesno(preferences.enable_memory_pooling));
    fprintf('Performance Monitoring: %s\n', yesno(preferences.enable_performance_monitoring));

    fprintf('\nSetup complete! Launch the GUI with: launch_enhanced_gui\n');
end

function result = yesno(condition)
    % YESNO - Convert boolean to Yes/No string
    if condition
        result = 'Yes';
    else
        result = 'No';
    end
end

% Alternative: Quick configuration functions
function enable_all_optimizations()
    % ENABLE_ALL_OPTIMIZATIONS - Enable all performance optimizations
    script_dir = fileparts(mfilename('fullpath'));
    pref_file = fullfile(script_dir, 'user_preferences.mat');

    if exist(pref_file, 'file')
        loaded_prefs = load(pref_file);
        preferences = loaded_prefs.preferences;
    else
        preferences = struct();
    end

    % Enable all optimizations
    preferences.enable_preallocation = true;
    preferences.enable_model_caching = true;
    preferences.enable_parallel_processing = true;
    preferences.enable_data_compression = true;
    preferences.enable_memory_pooling = true;
    preferences.enable_performance_monitoring = true;
    preferences.enable_memory_monitoring = true;

    save(pref_file, 'preferences');
    fprintf('All performance optimizations enabled!\n');
end

function disable_all_optimizations()
    % DISABLE_ALL_OPTIMIZATIONS - Disable all performance optimizations
    script_dir = fileparts(mfilename('fullpath'));
    pref_file = fullfile(script_dir, 'user_preferences.mat');

    if exist(pref_file, 'file')
        loaded_prefs = load(pref_file);
        preferences = loaded_prefs.preferences;
    else
        preferences = struct();
    end

    % Disable all optimizations
    preferences.enable_preallocation = false;
    preferences.enable_model_caching = false;
    preferences.enable_parallel_processing = false;
    preferences.enable_data_compression = false;
    preferences.enable_memory_pooling = false;
    preferences.enable_performance_monitoring = false;
    preferences.enable_memory_monitoring = false;

    save(pref_file, 'preferences');
    fprintf('All performance optimizations disabled!\n');
end

function show_current_preferences()
    % SHOW_CURRENT_PREFERENCES - Display current preference settings
    script_dir = fileparts(mfilename('fullpath'));
    pref_file = fullfile(script_dir, 'user_preferences.mat');

    if exist(pref_file, 'file')
        loaded_prefs = load(pref_file);
        preferences = loaded_prefs.preferences;

        fprintf('Current Performance Preferences:\n');
        fprintf('================================\n');

        % Performance optimizations
        fprintf('Preallocation: %s\n', yesno(isfield(preferences, 'enable_preallocation') && preferences.enable_preallocation));
        fprintf('Model Caching: %s\n', yesno(isfield(preferences, 'enable_model_caching') && preferences.enable_model_caching));
        fprintf('Parallel Processing: %s\n', yesno(isfield(preferences, 'enable_parallel_processing') && preferences.enable_parallel_processing));
        fprintf('Data Compression: %s\n', yesno(isfield(preferences, 'enable_data_compression') && preferences.enable_data_compression));
        fprintf('Memory Pooling: %s\n', yesno(isfield(preferences, 'enable_memory_pooling') && preferences.enable_memory_pooling));
        fprintf('Performance Monitoring: %s\n', yesno(isfield(preferences, 'enable_performance_monitoring') && preferences.enable_performance_monitoring));

        % Settings
        if isfield(preferences, 'max_parallel_workers')
            fprintf('Max Parallel Workers: %d\n', preferences.max_parallel_workers);
        end
        if isfield(preferences, 'cluster_profile')
            fprintf('Cluster Profile: %s\n', preferences.cluster_profile);
        end
        if isfield(preferences, 'use_local_cluster')
            fprintf('Use Local Cluster: %s\n', yesno(preferences.use_local_cluster));
        end
        if isfield(preferences, 'compression_level')
            fprintf('Compression Level: %d\n', preferences.compression_level);
        end
        if isfield(preferences, 'memory_pool_size')
            fprintf('Memory Pool Size: %d MB\n', preferences.memory_pool_size);
        end

        % Last used files
        if isfield(preferences, 'last_model_name') && ~isempty(preferences.last_model_name)
            fprintf('Last Model: %s\n', preferences.last_model_name);
        end
        if isfield(preferences, 'last_input_file') && ~isempty(preferences.last_input_file)
            fprintf('Last Input File: %s\n', preferences.last_input_file);
        end

    else
        fprintf('No preferences file found. Run setup_performance_preferences() to create one.\n');
    end
end
