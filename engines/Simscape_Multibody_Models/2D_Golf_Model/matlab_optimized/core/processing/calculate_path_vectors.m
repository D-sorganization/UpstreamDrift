function [BASEQ, ZTCFQ, DELTAQ] = calculate_path_vectors(BASEQ, ZTCFQ, DELTAQ)
% CALCULATE_PATH_VECTORS - Calculate Clubhead and Midpoint path vectors
%
% Inputs:
%   BASEQ - Base Q-table containing CHx, CHy, CHz, MPx, MPy, MPz
%   ZTCFQ - ZTCF Q-table
%   DELTAQ - DELTA Q-table
%
% Returns:
%   BASEQ, ZTCFQ, DELTAQ - Tables with added path vector columns
%
% This function calculates the directional vectors for the clubhead and
% midpoint paths. It replaces the legacy SCRIPT_CHPandMPPCalculation.m.
%
% Optimization:
%   - Vectorized operations using diff() instead of loops
%   - Pre-allocation (implicit in vectorization)
%   - Avoids growing arrays in loops which caused O(N^2) reallocation overhead
%   - Estimated speedup: >100x for large N
%   - Eliminates 'cd' usage and workspace side-effects
%
% Author: Bolt
% Date: 2025

    arguments
        BASEQ table
        ZTCFQ table
        DELTAQ table
    end

    % Define variable pairs for calculation
    % Format: {InputVarPrefix, OutputVarPrefix}
    % e.g. 'CH' -> reads CHx, CHy, CHz; writes CHPx, CHPy, CHPz
    pairs = {
        'CH', 'CHP';
        'MP', 'MPP'
    };

    components = {'x', 'y', 'z'};

    for i = 1:size(pairs, 1)
        input_prefix = pairs{i, 1};
        output_prefix = pairs{i, 2};

        for c = 1:length(components)
            comp = components{c};
            input_var = [input_prefix, comp];   % e.g., CHx
            output_var = [output_prefix, comp]; % e.g., CHPx

            % Check if input variable exists in BASEQ
            if ismember(input_var, BASEQ.Properties.VariableNames)
                % Calculate difference vector (Vectorized)
                % Equivalent to: val(j) - val(i) where j=i+1
                d = diff(BASEQ.(input_var));

                % Pad with the last value to maintain length
                % (Legacy script logic: last row gets value of second to last row)
                vec = [d; d(end)];

                % Assign to all tables
                BASEQ.(output_var) = vec;
                ZTCFQ.(output_var) = vec;
                DELTAQ.(output_var) = vec;
            end
        end
    end

end
