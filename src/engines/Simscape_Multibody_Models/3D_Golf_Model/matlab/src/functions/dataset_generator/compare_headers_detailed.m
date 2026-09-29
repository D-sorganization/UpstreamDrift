% Compare headers between 1956 and 1809 column datasets
fprintf('Comparing CSV headers to identify missing columns...\n');

% Read headers
fid = fopen('1956_headers.txt', 'r');
header_1956 = fgetl(fid);
fclose(fid);

fid = fopen('1809_headers.txt', 'r');
header_1809 = fgetl(fid);
fclose(fid);

% Split into cell arrays
headers_1956 = strsplit(header_1956, ',');
headers_1809 = strsplit(header_1809, ',');

fprintf('1956 columns: %d\n', length(headers_1956));
fprintf('1809 columns: %d\n', length(headers_1809));
fprintf('Missing columns: %d\n', length(headers_1956) - length(headers_1809));

% Find missing columns
missing_columns = setdiff(headers_1956, headers_1809);
extra_columns = setdiff(headers_1809, headers_1956);

fprintf('\n=== MISSING COLUMNS (%d) ===\n', length(missing_columns));
for i = 1:length(missing_columns)
    fprintf('%s\n', missing_columns{i});
end

fprintf('\n=== EXTRA COLUMNS (%d) ===\n', length(extra_columns));
for i = 1:length(extra_columns)
    fprintf('%s\n', extra_columns{i});
end

% Analyze patterns in missing columns
fprintf('\n=== PATTERN ANALYSIS ===\n');
missing_patterns = {};
for i = 1:length(missing_columns)
    col = missing_columns{i};

    % Extract base name (before last underscore)
    parts = strsplit(col, '_');
    if length(parts) > 1
        base_name = strjoin(parts(1:end-1), '_');
    else
        base_name = col;
    end

    if ~ismember(base_name, missing_patterns)
        missing_patterns{end+1} = base_name;
    end
end

fprintf('Unique base patterns in missing columns:\n');
for i = 1:length(missing_patterns)
    fprintf('  %s\n', missing_patterns{i});
end

% Save results
fid = fopen('missing_columns_analysis.txt', 'w');
fprintf(fid, 'MISSING COLUMNS ANALYSIS\n');
fprintf(fid, '=======================\n\n');
fprintf(fid, 'Total missing: %d\n', length(missing_columns));
fprintf(fid, 'Total extra: %d\n', length(extra_columns));
fprintf(fid, '\nMISSING COLUMNS:\n');
for i = 1:length(missing_columns)
    fprintf(fid, '%s\n', missing_columns{i});
end
fclose(fid);

fprintf('\nAnalysis complete. Results saved to missing_columns_analysis.txt\n');
