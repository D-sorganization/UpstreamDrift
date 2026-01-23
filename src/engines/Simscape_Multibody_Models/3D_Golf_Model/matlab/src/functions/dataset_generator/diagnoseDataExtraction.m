function diagnoseDataExtraction(simOut, config)
% DIAGNOSEDATAEXTRACTION - Diagnostic function to check data extraction
% This function analyzes what data sources are available and what's being extracted
% to help identify why we're not getting 1956 columns

fprintf('\n=== DATA EXTRACTION DIAGNOSTIC ===\n');

% Check what's available in simOut
fprintf('\n1. SIMULATION OUTPUT ANALYSIS:\n');
fprintf('   simOut class: %s\n', class(simOut));

% Check all properties of simOut
if isobject(simOut)
    props = properties(simOut);
    fprintf('   Available properties: %s\n', strjoin(props, ', '));

    for i = 1:length(props)
        prop_name = props{i};
        try
            prop_value = simOut.(prop_name);
            if isempty(prop_value)
                fprintf('   - %s: EMPTY\n', prop_name);
            else
                fprintf('   - %s: %s (size: %s)\n', prop_name, class(prop_value), mat2str(size(prop_value)));
            end
        catch ME
            fprintf('   - %s: ERROR accessing - %s\n', prop_name, ME.message);
        end
    end
end

% Check specific data sources
fprintf('\n2. DATA SOURCE ANALYSIS:\n');

% CombinedSignalBus
if isprop(simOut, 'CombinedSignalBus') || isfield(simOut, 'CombinedSignalBus')
    try
        combinedBus = simOut.CombinedSignalBus;
        if ~isempty(combinedBus)
            fprintf('   ✓ CombinedSignalBus: AVAILABLE\n');
            if isstruct(combinedBus)
                bus_fields = fieldnames(combinedBus);
                fprintf('     Fields: %s\n', strjoin(bus_fields, ', '));
                fprintf('     Field count: %d\n', length(bus_fields));
            end
        else
            fprintf('   ✗ CombinedSignalBus: EMPTY\n');
        end
    catch ME
        fprintf('   ✗ CombinedSignalBus: ERROR - %s\n', ME.message);
    end
else
    fprintf('   ✗ CombinedSignalBus: NOT AVAILABLE\n');
end

% Logsout
if isprop(simOut, 'logsout') || isfield(simOut, 'logsout')
    try
        logsout_data = simOut.logsout;
        if ~isempty(logsout_data)
            fprintf('   ✓ Logsout: AVAILABLE\n');
            if isstruct(logsout_data)
                logsout_fields = fieldnames(logsout_data);
                fprintf('     Fields: %s\n', strjoin(logsout_fields, ', '));
                fprintf('     Field count: %d\n', length(logsout_fields));
            end
        else
            fprintf('   ✗ Logsout: EMPTY\n');
        end
    catch ME
        fprintf('   ✗ Logsout: ERROR - %s\n', ME.message);
    end
else
    fprintf('   ✗ Logsout: NOT AVAILABLE\n');
end

% Simscape simlog
if isprop(simOut, 'simlog') || isfield(simOut, 'simlog')
    try
        simlog_data = simOut.simlog;
        if ~isempty(simlog_data)
            fprintf('   ✓ Simscape simlog: AVAILABLE\n');
            fprintf('     Class: %s\n', class(simlog_data));
            if isstruct(simlog_data)
                simlog_fields = fieldnames(simlog_data);
                fprintf('     Fields: %s\n', strjoin(simlog_fields, ', '));
                fprintf('     Field count: %d\n', length(simlog_fields));
            end
        else
            fprintf('   ✗ Simscape simlog: EMPTY\n');
        end
    catch ME
        fprintf('   ✗ Simscape simlog: ERROR - %s\n', ME.message);
    end
else
    fprintf('   ✗ Simscape simlog: NOT AVAILABLE\n');
end

% Model workspace
try
    if isprop(simOut, 'SimulationMetadata')
        model_name = simOut.SimulationMetadata.ModelInfo.ModelName;
        fprintf('   ✓ Model name: %s\n', model_name);

        if bdIsLoaded(model_name)
            model_workspace = get_param(model_name, 'ModelWorkspace');
            try
                variables = model_workspace.getVariableNames;
                fprintf('   ✓ Model workspace variables: %d found\n', length(variables));
                if length(variables) > 0
                    fprintf('     Variables: %s\n', strjoin(variables(1:min(10, length(variables))), ', '));
                    if length(variables) > 10
                        fprintf('     ... and %d more\n', length(variables) - 10);
                    end
                end
            catch ME
                fprintf('   ✗ Model workspace: ERROR accessing variables - %s\n', ME.message);
            end
        else
            fprintf('   ✗ Model workspace: MODEL NOT LOADED\n');
        end
    end
catch ME
    fprintf('   ✗ Model workspace: ERROR - %s\n', ME.message);
end

% Test extraction with current functions
fprintf('\n3. EXTRACTION FUNCTION TESTING:\n');

% Test CombinedSignalBus extraction
if isprop(simOut, 'CombinedSignalBus') || isfield(simOut, 'CombinedSignalBus')
    try
        combinedBus = simOut.CombinedSignalBus;
        if ~isempty(combinedBus)
            signal_bus_data = extractFromCombinedSignalBus(combinedBus);
            if ~isempty(signal_bus_data)
                fprintf('   ✓ CombinedSignalBus extraction: %d columns\n', width(signal_bus_data));
            else
                fprintf('   ✗ CombinedSignalBus extraction: NO DATA\n');
            end
        end
    catch ME
        fprintf('   ✗ CombinedSignalBus extraction: ERROR - %s\n', ME.message);
    end
end

% Test logsout extraction
if isprop(simOut, 'logsout') || isfield(simOut, 'logsout')
    try
        logsout_data = simOut.logsout;
        if ~isempty(logsout_data)
            logsout_table = extractLogsoutDataFixed(logsout_data);
            if ~isempty(logsout_table)
                fprintf('   ✓ Logsout extraction: %d columns\n', width(logsout_table));
            else
                fprintf('   ✗ Logsout extraction: NO DATA\n');
            end
        end
    catch ME
        fprintf('   ✗ Logsout extraction: ERROR - %s\n', ME.message);
    end
end

% Test Simscape extraction
if isprop(simOut, 'simlog') || isfield(simOut, 'simlog')
    try
        simlog_data = simOut.simlog;
        if ~isempty(simlog_data)
            simscape_table = extractSimscapeDataRecursive(simlog_data);
            if ~isempty(simscape_table)
                fprintf('   ✓ Simscape extraction: %d columns\n', width(simscape_table));
            else
                fprintf('   ✗ Simscape extraction: NO DATA\n');
            end
        end
    catch ME
        fprintf('   ✗ Simscape extraction: ERROR - %s\n', ME.message);
    end
end

% Test full extraction
fprintf('\n4. FULL EXTRACTION TEST:\n');
try
    options = struct();
    options.extract_combined_bus = true;
    options.extract_logsout = true;
    options.extract_simscape = true;
    options.verbose = true;

    [data_table, signal_info] = extractSignalsFromSimOut(simOut, options);

    if ~isempty(data_table)
        fprintf('   ✓ Full extraction: %d rows, %d columns\n', height(data_table), width(data_table));
        fprintf('   ✓ Data sources found: %d\n', signal_info.sources_found);
        fprintf('   ✓ Total columns: %d\n', signal_info.total_columns);

        % Report column count status
        fprintf('   ✓ Total columns extracted: %d\n', width(data_table));
    else
        fprintf('   ✗ Full extraction: NO DATA\n');
    end
catch ME
    fprintf('   ✗ Full extraction: ERROR - %s\n', ME.message);
end

% Configuration check
fprintf('\n5. CONFIGURATION CHECK:\n');
fprintf('   use_signal_bus: %d\n', config.use_signal_bus);
fprintf('   use_logsout: %d\n', config.use_logsout);
fprintf('   use_simscape: %d\n', config.use_simscape);
fprintf('   capture_workspace: %d\n', config.capture_workspace);

fprintf('\n=== DIAGNOSTIC COMPLETE ===\n');
end
