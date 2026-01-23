function PostProcessingModule()
    % PostProcessingModule - Standalone module for processing golf swing data
    % This module can be used independently or integrated with the GUI

    % Initialize the module
    module = struct();
    module.data_folder = '';
    module.output_folder = '';
    module.export_format = 'CSV';
    module.batch_size = 50;
    module.generate_features = true;
    module.compress_data = false;
    module.include_metadata = true;
    module.progress_callback = @defaultProgressCallback;
    module.log_callback = @defaultLogCallback;

    % Store module in global workspace for access
    assignin('base', 'postProcessingModule', module);

    fprintf('PostProcessingModule initialized. Use processDataFolder() to start processing.\n');
end

function processDataFolder(data_folder, varargin)
    % Process all data files in a folder
    % Usage: processDataFolder('path/to/data', 'output_folder', 'path/to/output', 'format', 'CSV')

    % Parse input arguments
    p = inputParser;
    addRequired(p, 'data_folder', @ischar);
    addParameter(p, 'output_folder', '', @ischar);
    addParameter(p, 'format', 'CSV', @ischar);
    addParameter(p, 'batch_size', 50, @isnumeric);
    addParameter(p, 'generate_features', true, @islogical);
    addParameter(p, 'compress_data', false, @islogical);
    addParameter(p, 'include_metadata', true, @islogical);
    addParameter(p, 'progress_callback', @defaultProgressCallback, @isfunction);
    addParameter(p, 'log_callback', @defaultLogCallback, @isfunction);

    parse(p, data_folder, varargin{:});

    % Get parameters
    output_folder = p.Results.output_folder;
    if isempty(output_folder)
        output_folder = fullfile(data_folder, 'processed_data');
    end

    % Create output folder if it doesn't exist
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Get all .mat files in the data folder
    files = dir(fullfile(data_folder, '*.mat'));
    if isempty(files)
        error('No .mat files found in the specified folder');
    end

    file_names = {files.name};
    total_files = length(file_names);

    % Initialize feature list if requested
    if p.Results.generate_features
        feature_list = initializeFeatureList();
    end

    % Process files in batches
    batch_size = p.Results.batch_size;
    num_batches = ceil(total_files / batch_size);

    p.Results.progress_callback('Starting data processing...', 0);
    p.Results.log_callback(sprintf('Processing %d files in %d batches', total_files, num_batches));

    for batch_idx = 1:num_batches
        % Calculate batch indices
        start_idx = (batch_idx - 1) * batch_size + 1;
        end_idx = min(batch_idx * batch_size, total_files);
        batch_files = file_names(start_idx:end_idx);

        % Update progress
        progress_ratio = batch_idx / num_batches;
        p.Results.progress_callback(sprintf('Processing batch %d/%d...', batch_idx, num_batches), progress_ratio);

        % Process batch
        batch_data = processBatch(batch_files, data_folder, p.Results);

        % Export batch
        exportBatch(batch_data, batch_idx, output_folder, p.Results);

        % Update feature list
        if p.Results.generate_features
            feature_list = updateFeatureList(feature_list, batch_data);
        end

        % Log progress
        p.Results.log_callback(sprintf('Completed batch %d/%d (%d files)', batch_idx, num_batches, length(batch_files)));
    end

    % Finalize processing
    if p.Results.generate_features
        exportFeatureList(feature_list, output_folder);
        p.Results.log_callback('Feature list exported to feature_list.json');
    end

    p.Results.progress_callback('Processing complete!', 1.0);
    p.Results.log_callback(sprintf('Processing completed successfully. %d files processed in %d batches.', total_files, num_batches));
end

function batch_data = processBatch(batch_files, data_folder, options)
    % Process a batch of files
    batch_data = struct();
    batch_data.trials = cell(length(batch_files), 1);
    batch_data.metadata = cell(length(batch_files), 1);
    batch_data.file_names = batch_files;

    for i = 1:length(batch_files)
        file_path = fullfile(data_folder, batch_files{i});

        try
            % Load data
            data = load(file_path);

            % Process data
            processed_trial = processTrialData(data, options);

            % Extract metadata
            metadata = extractMetadata(data, batch_files{i}, options);

            batch_data.trials{i} = processed_trial;
            batch_data.metadata{i} = metadata;

        catch ME
            warning('Failed to process file %s: %s', batch_files{i}, ME.message);
            options.log_callback(sprintf('ERROR processing %s: %s', batch_files{i}, ME.message));
        end
    end

    % Remove empty trials
    valid_trials = ~cellfun(@isempty, batch_data.trials);
    batch_data.trials = batch_data.trials(valid_trials);
    batch_data.metadata = batch_data.metadata(valid_trials);
    batch_data.file_names = batch_data.file_names(valid_trials);
end

function processed_trial = processTrialData(data, options)
    % Process individual trial data
    processed_trial = struct();

    % Extract time series data
    if isfield(data, 'time')
        processed_trial.time = data.time;
    end

    % Extract joint data
    if isfield(data, 'joint_data')
        processed_trial.joint_data = data.joint_data;
    end

    % Extract force data
    if isfield(data, 'force_data')
        processed_trial.force_data = data.force_data;
    end

    % Extract torque data
    if isfield(data, 'torque_data')
        processed_trial.torque_data = data.torque_data;
    end

    % Extract position data
    if isfield(data, 'position_data')
        processed_trial.position_data = data.position_data;
    end

    % Extract velocity data
    if isfield(data, 'velocity_data')
        processed_trial.velocity_data = data.velocity_data;
    end

    % Extract acceleration data
    if isfield(data, 'acceleration_data')
        processed_trial.acceleration_data = data.acceleration_data;
    end

    % Extract Simscape data if available
    if isfield(data, 'simscape_data')
        processed_trial.simscape_data = data.simscape_data;
    end

    % Extract workspace data if available
    if isfield(data, 'workspace_data')
        processed_trial.workspace_data = data.workspace_data;
    end

    % Calculate derived quantities
    processed_trial = calculateDerivedQuantities(processed_trial);
end

function metadata = extractMetadata(data, filename, options)
    % Extract metadata from trial data
    metadata = struct();
    metadata.filename = filename;
    metadata.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    metadata.file_size = 0; % Will be calculated if needed

    % Extract simulation parameters
    if isfield(data, 'simulation_params')
        metadata.simulation_params = data.simulation_params;
    end

    % Extract model parameters
    if isfield(data, 'model_params')
        metadata.model_params = data.model_params;
    end

    % Extract processing options
    metadata.processing_options = options;
end

function processed_trial = calculateDerivedQuantities(processed_trial)
    % Calculate derived quantities from raw data

    % Calculate angular velocities from joint angles
    if isfield(processed_trial, 'joint_data') && isfield(processed_trial, 'time')
        joint_names = fieldnames(processed_trial.joint_data);
        for i = 1:length(joint_names)
            joint_name = joint_names{i};
            if isfield(processed_trial.joint_data.(joint_name), 'angle')
                angle_data = processed_trial.joint_data.(joint_name).angle;
                time_data = processed_trial.time;

                % Calculate angular velocity using finite differences
                if length(angle_data) > 1
                    angular_velocity = diff(angle_data) ./ diff(time_data);
                    % Pad with zeros to match original length
                    angular_velocity = [angular_velocity; 0];
                    processed_trial.joint_data.(joint_name).angular_velocity = angular_velocity;
                end
            end
        end
    end

    % Calculate linear velocities from positions
    if isfield(processed_trial, 'position_data') && isfield(processed_trial, 'time')
        position_names = fieldnames(processed_trial.position_data);
        for i = 1:length(position_names)
            pos_name = position_names{i};
            if isfield(processed_trial.position_data.(pos_name), 'x')
                x_data = processed_trial.position_data.(pos_name).x;
                y_data = processed_trial.position_data.(pos_name).y;
                z_data = processed_trial.position_data.(pos_name).z;
                time_data = processed_trial.time;

                % Calculate velocities
                if length(x_data) > 1
                    % Calculate and pad with zeros (vectorized for performance)
                    vx = [diff(x_data) ./ diff(time_data); 0];
                    vy = [diff(y_data) ./ diff(time_data); 0];
                    vz = [diff(z_data) ./ diff(time_data); 0];

                    processed_trial.position_data.(pos_name).velocity_x = vx;
                    processed_trial.position_data.(pos_name).velocity_y = vy;
                    processed_trial.position_data.(pos_name).velocity_z = vz;

                    % Calculate speed magnitude
                    speed = sqrt(vx.^2 + vy.^2 + vz.^2);
                    processed_trial.position_data.(pos_name).speed = speed;
                end
            end
        end
    end

    % Calculate work and power with granular angular impulse
    if isfield(processed_trial, 'torque_data') && isfield(processed_trial, 'joint_data')
        % Default options - work calculations disabled for random input data
        calculation_options = struct();
        calculation_options.calculate_work = false;  % Disable work for random inputs

        processed_trial = calculateWorkAndPowerEnhanced(processed_trial, calculation_options);
    end
end

function processed_trial = calculateWorkAndPowerEnhanced(processed_trial, options)
    % Enhanced work and power calculation with granular angular impulse
    % Calculate work and power from torque and angular velocity data

    % Set default options
    if nargin < 2
        options = struct();
    end

    % Set defaults for all calculation options
    default_options = {
        'calculate_work', false;
        'calculate_power', true;
        'calculate_joint_torque_impulse', true;
        'calculate_applied_torque_impulse', true;
        'calculate_force_moment_impulse', true;
        'calculate_total_angular_impulse', true;
        'calculate_linear_impulse', true;
        'calculate_proximal_on_distal', true;
        'calculate_distal_on_proximal', true;


    };

    for i = 1:size(default_options, 1)
        if ~isfield(options, default_options{i, 1})
            options.(default_options{i, 1}) = default_options{i, 2};
        end
    end

    % Extract ZTCFQ and DELTAQ tables if available
    if isfield(processed_trial, 'ZTCFQ') && isfield(processed_trial, 'DELTAQ')
        ZTCFQ = processed_trial.ZTCFQ;
        DELTAQ = processed_trial.DELTAQ;

        % Call the enhanced granular calculation function
        [ZTCFQ_updated, DELTAQ_updated] = calculateWorkPowerAndGranularAngularImpulse3D(ZTCFQ, DELTAQ, options);

        % Update the processed trial with new data
        processed_trial.ZTCFQ = ZTCFQ_updated;
        processed_trial.DELTAQ = DELTAQ_updated;
    else
        % Fallback to original calculation method if ZTCFQ/DELTAQ not available
        joint_names = fieldnames(processed_trial.joint_data);
        torque_names = fieldnames(processed_trial.torque_data);

        total_work = 0;
        total_power = 0;

        for i = 1:length(joint_names)
            joint_name = joint_names{i};

            % Find corresponding torque data
            torque_idx = find(strcmp(torque_names, joint_name));
            if ~isempty(torque_idx)
                torque_name = torque_names{torque_idx};

                if isfield(processed_trial.joint_data.(joint_name), 'angular_velocity') && ...
                   isfield(processed_trial.torque_data.(torque_name), 'torque')

                    angular_velocity = processed_trial.joint_data.(joint_name).angular_velocity;
                    torque = processed_trial.torque_data.(torque_name).torque;

                    % Calculate power (P = τ * ω)
                    power = torque .* angular_velocity;
                    processed_trial.joint_data.(joint_name).power = power;

                    % Calculate work if requested (W = ∫ P dt)
                    if options.calculate_work && isfield(processed_trial, 'time')
                        time_data = processed_trial.time;
                        work = trapz(time_data, power);
                        processed_trial.joint_data.(joint_name).work = work;
                        total_work = total_work + work;
                    end

                    % Calculate peak power
                    peak_power = max(abs(power));
                    processed_trial.joint_data.(joint_name).peak_power = peak_power;
                    total_power = total_power + peak_power;
                end
            end
        end

        % Store totals
        processed_trial.total_peak_power = total_power;

        if options.calculate_work
            processed_trial.total_work = total_work;
        end
    end

    % Store calculation options for reference
    processed_trial.calculation_options = options;

    % Note: Granular angular impulse calculations are handled by
    % calculateWorkPowerAndGranularAngularImpulse3D function which provides:
    % - Joint torque angular impulse (proximal/distal)
    % - Applied torque angular impulse (proximal/distal)
    % - Force moment angular impulse (proximal/distal)
    % - Total angular impulse per joint end
    % - Linear impulse from joint forces
end

function exportBatch(batch_data, batch_idx, output_folder, options)
    % Export batch data in specified format
    output_file = fullfile(output_folder, sprintf('batch_%03d.%s', batch_idx, lower(options.format)));

    switch lower(options.format)
        case 'csv'
            exportToCSV(batch_data, output_file, options);
        case 'parquet'
            exportToParquet(batch_data, output_file, options);
        case 'mat'
            exportToMAT(batch_data, output_file, options);
        case 'json'
            exportToJSON(batch_data, output_file, options);
        case 'pytorch (.pt)'
            exportToPyTorch(batch_data, output_file, options);
        case 'tensorflow (.h5)'
            exportToTensorFlow(batch_data, output_file, options);
        case 'numpy (.npz)'
            exportToNumPy(batch_data, output_file, options);
        case 'pickle (.pkl)'
            exportToPickle(batch_data, output_file, options);
        otherwise
            error('Unsupported export format: %s', options.format);
    end
end

function exportToCSV(batch_data, output_file, options)
    % Export to CSV format
    % This is a simplified version - in practice, you'd need to flatten the nested structure

    % Create a flattened table for CSV export
    csv_data = struct();

    for i = 1:length(batch_data.trials)
        trial = batch_data.trials{i};
        trial_prefix = sprintf('trial_%d_', i);

        % Flatten trial data
        csv_data = flattenStruct(trial, csv_data, trial_prefix);
    end

    % Convert to table and save
    if ~isempty(fieldnames(csv_data))
        % Create a simple table with time series data
        time_data = [];
        if isfield(batch_data.trials{1}, 'time')
            time_data = batch_data.trials{1}.time;
        end

        % For now, save a simplified version
        % In practice, you'd need to handle the complex nested structure
        warning('CSV export simplified - complex nested data structures may not export completely');

        % Save basic time series data
        if ~isempty(time_data)
            writematrix(time_data, output_file);
        end
    end
end

function exportToParquet(batch_data, output_file, options)
    % Export to Parquet format
    % Note: This requires the Apache Arrow library or similar

    try
        % Convert to table format
        table_data = structToTable(batch_data);

        % Save as parquet (this is a placeholder - actual implementation depends on available libraries)
        warning('Parquet export not fully implemented - saving as MAT file instead');
        save(strrep(output_file, '.parquet', '.mat'), '-struct', 'batch_data');

    catch ME
        warning('Parquet export failed: %s. Falling back to MAT format.', ME.message);
        save(strrep(output_file, '.parquet', '.mat'), '-struct', 'batch_data');
    end
end

function exportToMAT(batch_data, output_file, options)
    % Export to MAT format
    if options.compress_data
        save(output_file, '-struct', 'batch_data', '-v7.3', '-nocompression');
    else
        save(output_file, '-struct', 'batch_data', '-v7.3');
    end
end

function exportToJSON(batch_data, output_file, options)
    % Export to JSON format
    % Note: This may not handle all MATLAB data types perfectly

    try
        % Convert to JSON-compatible format
        json_data = structToJSON(batch_data);

        % Write to file
        json_str = jsonencode(json_data, 'PrettyPrint', true);
        fid = fopen(output_file, 'w');
        fprintf(fid, '%s', json_str);
        fclose(fid);

    catch ME
        warning('JSON export failed: %s. Falling back to MAT format.', ME.message);
        save(strrep(output_file, '.json', '.mat'), '-struct', 'batch_data');
    end
end

function feature_list = initializeFeatureList()
    % Initialize feature list for machine learning
    feature_list = struct();
    feature_list.features = {};
    feature_list.descriptions = {};
    feature_list.units = {};
    feature_list.ranges = {};
    feature_list.categories = {};
    feature_list.feature_data = struct();
end

function feature_list = updateFeatureList(feature_list, batch_data)
    % Update feature list with new data
    % Extract features from batch data

    for i = 1:length(batch_data.trials)
        trial = batch_data.trials{i};

        % Extract joint features
        if isfield(trial, 'joint_data')
            feature_list = extractJointFeatures(feature_list, trial.joint_data, i);
        end

        % Extract position features
        if isfield(trial, 'position_data')
            feature_list = extractPositionFeatures(feature_list, trial.position_data, i);
        end

        % Extract force features
        if isfield(trial, 'force_data')
            feature_list = extractForceFeatures(feature_list, trial.force_data, i);
        end

        % Extract torque features
        if isfield(trial, 'torque_data')
            feature_list = extractTorqueFeatures(feature_list, trial.torque_data, i);
        end

        % Extract derived features
        if isfield(trial, 'total_work')
            feature_list = addFeature(feature_list, sprintf('trial_%d_total_work', i), ...
                                    trial.total_work, 'Total mechanical work', 'J', 'energy');
        end

        if isfield(trial, 'total_peak_power')
            feature_list = addFeature(feature_list, sprintf('trial_%d_total_peak_power', i), ...
                                    trial.total_peak_power, 'Total peak power', 'W', 'power');
        end
    end

    % Update feature ranges
    feature_list = updateFeatureRanges(feature_list);
end

function feature_list = extractJointFeatures(feature_list, joint_data, trial_idx)
    % Extract features from joint data
    joint_names = fieldnames(joint_data);

    for i = 1:length(joint_names)
        joint_name = joint_names{i};
        joint = joint_data.(joint_name);

        if isfield(joint, 'angle')
            % Range of motion
            rom = max(joint.angle) - min(joint.angle);
            feature_list = addFeature(feature_list, sprintf('trial_%d_%s_rom', trial_idx, joint_name), ...
                                    rom, sprintf('%s range of motion', joint_name), 'rad', 'kinematics');

            % Peak angular velocity
            if isfield(joint, 'angular_velocity')
                peak_velocity = max(abs(joint.angular_velocity));
                feature_list = addFeature(feature_list, sprintf('trial_%d_%s_peak_velocity', trial_idx, joint_name), ...
                                        peak_velocity, sprintf('%s peak angular velocity', joint_name), 'rad/s', 'kinematics');
            end
        end
    end
end

function feature_list = extractPositionFeatures(feature_list, position_data, trial_idx)
    % Extract features from position data
    position_names = fieldnames(position_data);

    for i = 1:length(position_names)
        pos_name = position_names{i};
        position = position_data.(pos_name);

        if isfield(position, 'x') && isfield(position, 'y') && isfield(position, 'z')
            % Total displacement
            displacement = sqrt((position.x(end) - position.x(1))^2 + ...
                               (position.y(end) - position.y(1))^2 + ...
                               (position.z(end) - position.z(1))^2);
            feature_list = addFeature(feature_list, sprintf('trial_%d_%s_displacement', trial_idx, pos_name), ...
                                    displacement, sprintf('%s total displacement', pos_name), 'm', 'kinematics');

            % Peak speed
            if isfield(position, 'speed')
                peak_speed = max(position.speed);
                feature_list = addFeature(feature_list, sprintf('trial_%d_%s_peak_speed', trial_idx, pos_name), ...
                                        peak_speed, sprintf('%s peak speed', pos_name), 'm/s', 'kinematics');
            end
        end
    end
end

function feature_list = extractForceFeatures(feature_list, force_data, trial_idx)
    % Extract features from force data
    force_names = fieldnames(force_data);

    for i = 1:length(force_names)
        force_name = force_names{i};
        force = force_data.(force_name);

        if isfield(force, 'magnitude')
            % Peak force
            peak_force = max(force.magnitude);
            feature_list = addFeature(feature_list, sprintf('trial_%d_%s_peak_force', trial_idx, force_name), ...
                                    peak_force, sprintf('%s peak force', force_name), 'N', 'dynamics');

            % Mean force
            mean_force = mean(force.magnitude);
            feature_list = addFeature(feature_list, sprintf('trial_%d_%s_mean_force', trial_idx, force_name), ...
                                    mean_force, sprintf('%s mean force', force_name), 'N', 'dynamics');
        end
    end
end

function feature_list = extractTorqueFeatures(feature_list, torque_data, trial_idx)
    % Extract features from torque data
    torque_names = fieldnames(torque_data);

    for i = 1:length(torque_names)
        torque_name = torque_names{i};
        torque = torque_data.(torque_name);

        if isfield(torque, 'torque')
            % Peak torque
            peak_torque = max(abs(torque.torque));
            feature_list = addFeature(feature_list, sprintf('trial_%d_%s_peak_torque', trial_idx, torque_name), ...
                                    peak_torque, sprintf('%s peak torque', torque_name), 'N⋅m', 'dynamics');

            % Mean torque
            mean_torque = mean(abs(torque.torque));
            feature_list = addFeature(feature_list, sprintf('trial_%d_%s_mean_torque', trial_idx, torque_name), ...
                                    mean_torque, sprintf('%s mean torque', torque_name), 'N⋅m', 'dynamics');
        end
    end
end

function feature_list = addFeature(feature_list, name, value, description, unit, category)
    % Add a feature to the feature list
    feature_list.features{end+1} = name;
    feature_list.descriptions{end+1} = description;
    feature_list.units{end+1} = unit;
    feature_list.categories{end+1} = category;

    % Store the actual value
    feature_list.feature_data.(name) = value;
end

function feature_list = updateFeatureRanges(feature_list)
    % Update feature ranges based on collected data
    feature_names = fieldnames(feature_list.feature_data);

    for i = 1:length(feature_names)
        feature_name = feature_names{i};
        values = feature_list.feature_data.(feature_name);

        if isnumeric(values)
            if isscalar(values)
                feature_list.ranges{end+1} = [values, values];
            else
                feature_list.ranges{end+1} = [min(values), max(values)];
            end
        else
            feature_list.ranges{end+1} = [];
        end
    end
end

function exportFeatureList(feature_list, output_folder)
    % Export feature list for Python/ML use
    feature_file = fullfile(output_folder, 'feature_list.json');

    % Convert to JSON-compatible format
    feature_data = struct();
    feature_data.features = feature_list.features;
    feature_data.descriptions = feature_list.descriptions;
    feature_data.units = feature_list.units;
    feature_data.ranges = feature_list.ranges;
    feature_data.categories = feature_list.categories;
    feature_data.feature_data = feature_list.feature_data;

    % Add metadata
    feature_data.metadata = struct();
    feature_data.metadata.export_timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    feature_data.metadata.total_features = length(feature_list.features);
    feature_data.metadata.categories = unique(feature_list.categories);

    % Write to JSON file
    feature_json = jsonencode(feature_data, 'PrettyPrint', true);
    fid = fopen(feature_file, 'w');
    fprintf(fid, '%s', feature_json);
    fclose(fid);

    % Also create a Python-compatible CSV file
    csv_file = fullfile(output_folder, 'feature_data.csv');
    createFeatureCSV(feature_list, csv_file);
end

function createFeatureCSV(feature_list, csv_file)
    % Create a CSV file with feature data for Python
    feature_names = fieldnames(feature_list.feature_data);

    if ~isempty(feature_names)
        % Create table
        data_table = table();

        for i = 1:length(feature_names)
            feature_name = feature_names{i};
            values = feature_list.feature_data.(feature_name);

            if isnumeric(values)
                data_table.(feature_name) = values;
            else
                data_table.(feature_name) = {values};
            end
        end

        % Write to CSV
        writetable(data_table, csv_file);
    end
end

% Utility functions
function flattened = flattenStruct(data, flattened, prefix)
    % Flatten a nested structure for CSV export
    if isstruct(data)
        fields = fieldnames(data);
        for i = 1:length(fields)
            field_name = fields{i};
            field_data = data.(field_name);
            new_prefix = [prefix, field_name, '_'];
            flattened = flattenStruct(field_data, flattened, new_prefix);
        end
    else
        % Store the data
        field_name = prefix(1:end-1); % Remove trailing underscore
        flattened.(field_name) = data;
    end
end

function table_data = structToTable(data)
    % Convert structure to table format
    % This is a simplified version
    table_data = struct();

    % For now, just convert basic fields
    if isfield(data, 'trials')
        for i = 1:length(data.trials)
            trial = data.trials{i};
            if isfield(trial, 'time')
                table_data.(sprintf('trial_%d_time', i)) = trial.time;
            end
        end
    end

    table_data = struct2table(table_data);
end

function json_data = structToJSON(data)
    % Convert structure to JSON-compatible format
    json_data = data;

    % Handle special MATLAB types
    if isfield(json_data, 'trials')
        for i = 1:length(json_data.trials)
            trial = json_data.trials{i};
            if isstruct(trial)
                json_data.trials{i} = convertStructForJSON(trial);
            end
        end
    end
end

function converted = convertStructForJSON(data)
    % Convert structure fields to JSON-compatible format
    converted = struct();
    fields = fieldnames(data);

    for i = 1:length(fields)
        field_name = fields{i};
        field_data = data.(field_name);

        if isnumeric(field_data)
            converted.(field_name) = field_data;
        elseif ischar(field_data)
            converted.(field_name) = field_data;
        elseif iscell(field_data)
            converted.(field_name) = field_data;
        elseif isstruct(field_data)
            converted.(field_name) = convertStructForJSON(field_data);
        else
            converted.(field_name) = char(field_data);
        end
    end
end

% Default callback functions
function defaultProgressCallback(message, progress)
    % Default progress callback
    fprintf('Progress: %.1f%% - %s\n', progress * 100, message);
end

function defaultLogCallback(message)
    % Default log callback
    fprintf('[%s] %s\n', datestr(now, 'HH:MM:SS'), message);
end

% Machine Learning Export Functions
function exportToPyTorch(batch_data, output_file, options)
    % Export to PyTorch format (.pt)
    try
        % Convert data to PyTorch-compatible format
        pytorch_data = convertToPyTorchFormat(batch_data);

        % Save using Python interface (requires Python with PyTorch)
        if exist('pyenv', 'builtin')
            try
                % Create Python script to save PyTorch tensor
                python_script = sprintf([...
                    'import torch\n', ...
                    'import numpy as np\n', ...
                    'data = %s\n', ...
                    'torch.save(data, "%s")\n'], ...
                    matlab2json(pytorch_data), output_file);

                % Execute Python script
                system(sprintf('python -c "%s"', python_script));
                fprintf('PyTorch data saved to: %s\n', output_file);
            catch ME
                warning('PyTorch export failed: %s. Falling back to MAT format.', ME.message);
                save(strrep(output_file, '.pt', '.mat'), '-struct', 'batch_data');
            end
        else
            warning('Python interface not available. Falling back to MAT format.');
            save(strrep(output_file, '.pt', '.mat'), '-struct', 'batch_data');
        end
    catch ME
        warning('PyTorch export failed: %s. Falling back to MAT format.', ME.message);
        save(strrep(output_file, '.pt', '.mat'), '-struct', 'batch_data');
    end
end

function exportToTensorFlow(batch_data, output_file, options)
    % Export to TensorFlow format (.h5)
    try
        % Convert data to TensorFlow-compatible format
        tf_data = convertToTensorFlowFormat(batch_data);

        % Save using Python interface (requires Python with TensorFlow)
        if exist('pyenv', 'builtin')
            try
                % Create Python script to save TensorFlow data
                python_script = sprintf([...
                    'import h5py\n', ...
                    'import numpy as np\n', ...
                    'data = %s\n', ...
                    'with h5py.File("%s", "w") as f:\n', ...
                    '    for key, value in data.items():\n', ...
                    '        f.create_dataset(key, data=value)\n'], ...
                    matlab2json(tf_data), output_file);

                % Execute Python script
                system(sprintf('python -c "%s"', python_script));
                fprintf('TensorFlow data saved to: %s\n', output_file);
            catch ME
                warning('TensorFlow export failed: %s. Falling back to MAT format.', ME.message);
                save(strrep(output_file, '.h5', '.mat'), '-struct', 'batch_data');
            end
        else
            warning('Python interface not available. Falling back to MAT format.');
            save(strrep(output_file, '.h5', '.mat'), '-struct', 'batch_data');
        end
    catch ME
        warning('TensorFlow export failed: %s. Falling back to MAT format.', ME.message);
        save(strrep(output_file, '.h5', '.mat'), '-struct', 'batch_data');
    end
end

function exportToNumPy(batch_data, output_file, options)
    % Export to NumPy format (.npz)
    try
        % Convert data to NumPy-compatible format
        numpy_data = convertToNumPyFormat(batch_data);

        % Save using Python interface (requires Python with NumPy)
        if exist('pyenv', 'builtin')
            try
                % Create Python script to save NumPy data
                python_script = sprintf([...
                    'import numpy as np\n', ...
                    'data = %s\n', ...
                    'np.savez("%s", **data)\n'], ...
                    matlab2json(numpy_data), output_file);

                % Execute Python script
                system(sprintf('python -c "%s"', python_script));
                fprintf('NumPy data saved to: %s\n', output_file);
            catch ME
                warning('NumPy export failed: %s. Falling back to MAT format.', ME.message);
                save(strrep(output_file, '.npz', '.mat'), '-struct', 'batch_data');
            end
        else
            warning('Python interface not available. Falling back to MAT format.');
            save(strrep(output_file, '.npz', '.mat'), '-struct', 'batch_data');
        end
    catch ME
        warning('NumPy export failed: %s. Falling back to MAT format.', ME.message);
        save(strrep(output_file, '.npz', '.mat'), '-struct', 'batch_data');
    end
end

function exportToPickle(batch_data, output_file, options)
    % Export to Pickle format (.pkl)
    try
        % Convert data to Pickle-compatible format
        pickle_data = convertToPickleFormat(batch_data);

        % Save using Python interface (requires Python)
        if exist('pyenv', 'builtin')
            try
                % Create Python script to save Pickle data
                python_script = sprintf([...
                    'import pickle\n', ...
                    'import numpy as np\n', ...
                    'data = %s\n', ...
                    'with open("%s", "wb") as f:\n', ...
                    '    pickle.dump(data, f)\n'], ...
                    matlab2json(pickle_data), output_file);

                % Execute Python script
                system(sprintf('python -c "%s"', python_script));
                fprintf('Pickle data saved to: %s\n', output_file);
            catch ME
                warning('Pickle export failed: %s. Falling back to MAT format.', ME.message);
                save(strrep(output_file, '.pkl', '.mat'), '-struct', 'batch_data');
            end
        else
            warning('Python interface not available. Falling back to MAT format.');
            save(strrep(output_file, '.pkl', '.mat'), '-struct', 'batch_data');
        end
    catch ME
        warning('Pickle export failed: %s. Falling back to MAT format.', ME.message);
        save(strrep(output_file, '.pkl', '.mat'), '-struct', 'batch_data');
    end
end

% Data conversion functions for ML formats
function pytorch_data = convertToPyTorchFormat(batch_data)
    % Convert batch data to PyTorch-compatible format
    pytorch_data = struct();

    % Extract time series data
    if isfield(batch_data, 'trials') && ~isempty(batch_data.trials)
        trial = batch_data.trials{1}; % Use first trial as template

        % Convert to tensor format
        if isfield(trial, 'time')
            pytorch_data.time = trial.time;
        end

        % Extract joint data
        if isfield(trial, 'joint_data')
            joint_names = fieldnames(trial.joint_data);
            for i = 1:length(joint_names)
                joint_name = joint_names{i};
                joint_data = trial.joint_data.(joint_name);

                if isfield(joint_data, 'angular_velocity')
                    pytorch_data.([joint_name '_angular_velocity']) = joint_data.angular_velocity;
                end
                if isfield(joint_data, 'power')
                    pytorch_data.([joint_name '_power']) = joint_data.power;
                end
            end
        end

        % Extract torque data
        if isfield(trial, 'torque_data')
            torque_names = fieldnames(trial.torque_data);
            for i = 1:length(torque_names)
                torque_name = torque_names{i};
                torque_data = trial.torque_data.(torque_name);

                if isfield(torque_data, 'torque')
                    pytorch_data.([torque_name '_torque']) = torque_data.torque;
                end
            end
        end
    end
end

function tf_data = convertToTensorFlowFormat(batch_data)
    % Convert batch data to TensorFlow-compatible format
    tf_data = struct();

    % Similar to PyTorch but with TensorFlow-specific formatting
    if isfield(batch_data, 'trials') && ~isempty(batch_data.trials)
        trial = batch_data.trials{1};

        if isfield(trial, 'time')
            tf_data.time = trial.time;
        end

        % Extract features for TensorFlow
        if isfield(trial, 'joint_data')
            joint_names = fieldnames(trial.joint_data);
            for i = 1:length(joint_names)
                joint_name = joint_names{i};
                joint_data = trial.joint_data.(joint_name);

                if isfield(joint_data, 'angular_velocity')
                    tf_data.([joint_name '_angular_velocity']) = joint_data.angular_velocity;
                end
                if isfield(joint_data, 'power')
                    tf_data.([joint_name '_power']) = joint_data.power;
                end
            end
        end
    end
end

function numpy_data = convertToNumPyFormat(batch_data)
    % Convert batch data to NumPy-compatible format
    numpy_data = struct();

    % Extract all numeric data for NumPy
    if isfield(batch_data, 'trials') && ~isempty(batch_data.trials)
        trial = batch_data.trials{1};

        % Convert all numeric fields
        fields = fieldnames(trial);
        for i = 1:length(fields)
            field_name = fields{i};
            field_data = trial.(field_name);

            if isnumeric(field_data)
                numpy_data.(field_name) = field_data;
            elseif isstruct(field_data)
                % Recursively convert nested structures
                nested_data = convertStructToNumeric(field_data);
                if ~isempty(fieldnames(nested_data))
                    numpy_data.(field_name) = nested_data;
                end
            end
        end
    end
end

function pickle_data = convertToPickleFormat(batch_data)
    % Convert batch data to Pickle-compatible format
    pickle_data = struct();

    % Pickle can handle most Python-compatible data types
    if isfield(batch_data, 'trials') && ~isempty(batch_data.trials)
        trial = batch_data.trials{1};

        % Convert to Python-compatible format
        pickle_data = convertToPythonFormat(trial);
    end
end

function numeric_data = convertStructToNumeric(data)
    % Convert structure to numeric-only format
    numeric_data = struct();

    if isstruct(data)
        fields = fieldnames(data);
        for i = 1:length(fields)
            field_name = fields{i};
            field_data = data.(field_name);

            if isnumeric(field_data)
                numeric_data.(field_name) = field_data;
            elseif isstruct(field_data)
                nested_data = convertStructToNumeric(field_data);
                if ~isempty(fieldnames(nested_data))
                    numeric_data.(field_name) = nested_data;
                end
            end
        end
    end
end

function python_data = convertToPythonFormat(data)
    % Convert MATLAB data to Python-compatible format
    python_data = struct();

    if isstruct(data)
        fields = fieldnames(data);
        for i = 1:length(fields)
            field_name = fields{i};
            field_data = data.(field_name);

            if isnumeric(field_data)
                python_data.(field_name) = field_data;
            elseif ischar(field_data)
                python_data.(field_name) = field_data;
            elseif iscell(field_data)
                python_data.(field_name) = field_data;
            elseif isstruct(field_data)
                python_data.(field_name) = convertToPythonFormat(field_data);
            else
                python_data.(field_name) = char(field_data);
            end
        end
    end
end

function json_str = matlab2json(data)
    % Convert MATLAB data to JSON string for Python
    json_str = jsonencode(data);
end
