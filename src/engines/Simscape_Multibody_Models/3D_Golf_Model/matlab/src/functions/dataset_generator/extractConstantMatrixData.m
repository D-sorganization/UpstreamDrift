% Extract constant matrix/vector data and replicate for time series
function [constant_signals] = extractConstantMatrixData(data_value, signal_name, reference_time)
    constant_signals = {};

    try
        if isstruct(data_value)
            % Extract numeric data from struct fields
            fields = fieldnames(data_value);
            for i = 1:length(fields)
                field_name = fields{i};
                field_value = data_value.(field_name);

                % Recursively process struct fields
                if isnumeric(field_value)
                    sub_signals = extractConstantMatrixData(field_value, sprintf('%s_%s', signal_name, field_name), reference_time);
                    constant_signals = [constant_signals, sub_signals];
                end
            end

        elseif isnumeric(data_value)
            % Process numeric data directly
            num_elements = numel(data_value);
            data_size = size(data_value);

            % Determine reference length (use 3006 as default if no reference provided)
            if isempty(reference_time)
                expected_length = 3006;  % Default length based on typical simulation
            else
                expected_length = length(reference_time);
            end

            if num_elements == 3
                % 3D VECTOR (e.g., COM position [x, y, z])
                vector_data = data_value(:);  % Ensure column vector
                dim_labels = {'x', 'y', 'z'};
                for dim = 1:3
                    replicated_data = repmat(vector_data(dim), expected_length, 1);
                    signal_name_full = matlab.lang.makeValidName(sprintf('%s_%s', signal_name, dim_labels{dim}));
                    constant_signals{end+1} = struct('name', signal_name_full, 'data', replicated_data);
                    fprintf('Debug: Added 3D vector %s (replicated %g for %d timesteps)\n', signal_name_full, vector_data(dim), expected_length);
                end

            elseif num_elements == 6
                % 6DOF DATA (e.g., pose/twist [x,y,z,rx,ry,rz])
                vector_data = data_value(:);  % Ensure column vector
                dof_labels = {'x', 'y', 'z', 'rx', 'ry', 'rz'};
                for dim = 1:6
                    replicated_data = repmat(vector_data(dim), expected_length, 1);
                    signal_name_full = matlab.lang.makeValidName(sprintf('%s_%s', signal_name, dof_labels{dim}));
                    constant_signals{end+1} = struct('name', signal_name_full, 'data', replicated_data);
                    fprintf('Debug: Added 6DOF data %s (replicated %g for %d timesteps)\n', signal_name_full, vector_data(dim), expected_length);
                end

            elseif num_elements == 9 && isequal(data_size, [3, 3])
                % 3x3 MATRIX (e.g., inertia matrix, rotation matrix)
                matrix_data = data_value;
                for row = 1:3
                    for col = 1:3
                        matrix_element = matrix_data(row, col);
                        replicated_data = repmat(matrix_element, expected_length, 1);
                        signal_name_full = matlab.lang.makeValidName(sprintf('%s_R%d%d', signal_name, row, col));
                        constant_signals{end+1} = struct('name', signal_name_full, 'data', replicated_data);
                        fprintf('Debug: Added 3x3 matrix %s (replicated %g for %d timesteps)\n', signal_name_full, matrix_element, expected_length);
                    end
                end

            elseif num_elements == 9 && ~isequal(data_size, [3, 3])
                % 9-ELEMENT VECTOR (flattened 3x3 matrix)
                vector_data = data_value(:);  % Ensure column vector
                for elem = 1:9
                    row = ceil(elem/3);
                    col = mod(elem-1, 3) + 1;
                    replicated_data = repmat(vector_data(elem), expected_length, 1);
                    signal_name_full = matlab.lang.makeValidName(sprintf('%s_I%d%d', signal_name, row, col));
                    constant_signals{end+1} = struct('name', signal_name_full, 'data', replicated_data);
                    fprintf('Debug: Added 9-element vector %s (replicated %g for %d timesteps)\n', signal_name_full, vector_data(elem), expected_length);
                end

            elseif num_elements == 1
                % SCALAR CONSTANT
                replicated_data = repmat(data_value, expected_length, 1);
                signal_name_full = matlab.lang.makeValidName(signal_name);
                constant_signals{end+1} = struct('name', signal_name_full, 'data', replicated_data);
                fprintf('Debug: Added scalar constant %s (replicated %g for %d timesteps)\n', signal_name_full, data_value, expected_length);

            else
                % UNSUPPORTED SIZE - log for debugging
                fprintf('Debug: Skipping %s (unsupported size [%s] - need 1, 3, 6, or 9 elements)\n', signal_name, num2str(data_size));
            end
        end

    catch ME
        fprintf('Debug: Error processing constant data %s: %s\n', signal_name, ME.message);
    end
end
