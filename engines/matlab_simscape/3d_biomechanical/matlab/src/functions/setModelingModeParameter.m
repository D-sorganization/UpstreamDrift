function setModelingModeParameter(inputFilePath, newValue)
% Modifies a parameter in a .m input file
% Specifically sets 'ModelingMode = ...' to a new value

    % Read file
    if ~isfile(inputFilePath)
        error('File not found: %s', inputFilePath);
    end

    lines = readlines(inputFilePath, TextType="string");

    % Find and update ModelingMode assignment
    modeLine = find(contains(lines, 'ModelingMode'), 1);
    if isempty(modeLine)
        error('ModelingMode not found in file.');
    end

    lines(modeLine) = sprintf('ModelingMode = %d;', newValue);

    % Write back to file
    writelines(lines, inputFilePath);
    fprintf('Updated ModelingMode to %d in %s\n', newValue, inputFilePath);
end
