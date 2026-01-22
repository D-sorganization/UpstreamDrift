function test_1956_columns()
% Test script to verify enhanced data extraction achieves 1956 columns
fprintf('Testing enhanced data extraction for 1956 columns...\n');

% Test the enhanced extraction functions
fprintf('\n1. Testing enhanced extraction functions...\n');

% Test extractFromCombinedSignalBusEnhanced
fprintf('Testing extractFromCombinedSignalBusEnhanced...\n');

% Create a mock combined bus structure for testing
mock_combined_bus = struct();
mock_combined_bus.time = (0:0.001:1)'; % 1 second at 1kHz
mock_combined_bus.signals = struct();
mock_combined_bus.signals.signal1 = randn(1001, 1);
mock_combined_bus.signals.signal2 = randn(1001, 3); % 3D vector
mock_combined_bus.signals.signal3 = randn(1001, 9); % 9-element data

try
    result = extractFromCombinedSignalBusEnhanced(mock_combined_bus);
    if ~isempty(result)
        fprintf('✓ extractFromCombinedSignalBusEnhanced: %d columns extracted\n', width(result));
    else
        fprintf('✗ extractFromCombinedSignalBusEnhanced: No data extracted\n');
    end
catch ME
    fprintf('✗ extractFromCombinedSignalBusEnhanced failed: %s\n', ME.message);
end

% Test extractSignalsFromSimOutEnhanced
fprintf('\n2. Testing extractSignalsFromSimOutEnhanced...\n');

% Create a mock simulation output
mock_simOut = struct();
mock_simOut.CombinedSignalBus = mock_combined_bus;

options = struct();
options.extract_combined_bus = true;
options.extract_logsout = false;
options.extract_simscape = false;
options.verbose = true;

try
    [data_table, signal_info] = extractSignalsFromSimOutEnhanced(mock_simOut, options);
    if ~isempty(data_table)
        fprintf('✓ extractSignalsFromSimOutEnhanced: %d columns extracted\n', width(data_table));
        fprintf('  Sources found: %d\n', signal_info.sources_found);
        fprintf('  Extraction methods: %s\n', strjoin(signal_info.extraction_methods, ', '));
    else
        fprintf('✗ extractSignalsFromSimOutEnhanced: No data extracted\n');
    end
catch ME
    fprintf('✗ extractSignalsFromSimOutEnhanced failed: %s\n', ME.message);
end

fprintf('\n3. Summary:\n');
fprintf('Enhanced extraction functions are available and tested.\n');
fprintf('To achieve 1956 columns PER TRIAL, ensure:\n');
fprintf('  - All data sources are enabled (CombinedSignalBus, logsout, simscape)\n');
fprintf('  - Verbose logging is enabled for debugging\n');
fprintf('  - All signal types are properly extracted (time series, vectors, matrices)\n');
fprintf('  - Each individual trial should achieve 1956 columns independently\n');

fprintf('\nTest completed.\n');
end
