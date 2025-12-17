function combined_table = combineDataSources(data_sources)
    % Combine multiple data tables into one
    combined_table = [];

    try
        if isempty(data_sources)
            return;
        end

        % Start with the first data source
        combined_table = data_sources{1};

        % Merge additional data sources
        % Alternative: Use Grok's cleaner mergeTables function if all sources have time:
        % combined_table = mergeTables(data_sources{:});

        for i = 2:length(data_sources)
            if ~isempty(data_sources{i})
                % Find common time column
                if ismember('time', combined_table.Properties.VariableNames) && ...
                   ismember('time', data_sources{i}.Properties.VariableNames)

                    % Merge on time column (same as Grok's mergeTables)
                    combined_table = outerjoin(combined_table, data_sources{i}, 'Keys', 'time', 'MergeKeys', true);
                else
                    % Robust fallback for edge cases without time columns
                    common_vars = intersect(combined_table.Properties.VariableNames, ...
                                          data_sources{i}.Properties.VariableNames);
                    if ~isempty(common_vars)
                        combined_table = [combined_table(:, common_vars); data_sources{i}(:, common_vars)];
                    end
                end
            end
        end

    catch ME
        fprintf('Error combining data sources: %s\n', ME.message);
    end
end
