% Compare two SimulationInput .mat files or a .mat file with a model workspace
clear; clc;
tol = 1e-6;  % numeric tolerance

% Prompt for first file
[file1, path1] = uigetfile({'*.mat;*.slx'}, 'Select the first input or model file');
if isequal(file1, 0), disp('Canceled'); return; end
full1 = fullfile(path1, file1);

% Prompt for second file
[file2, path2] = uigetfile({'*.mat;*.slx'}, 'Select the second input or model file');
if isequal(file2, 0), disp('Canceled'); return; end
full2 = fullfile(path2, file2);

% --- Load File 1 ---
if endsWith(full1, '.mat')
    s1 = load(full1);
    if isfield(s1, 'in') && isa(s1.in, 'Simulink.SimulationInput')
        vars1 = s1.in.Variables;
    else
        % Try to extract Simulink.Parameter entries
        names = fieldnames(s1);
        vars1 = struct('Name', {}, 'Value', {});
        for i = 1:length(names)
            val = s1.(names{i});
            if isa(val, 'Simulink.Parameter')
                vars1(end+1).Name = names{i};
                vars1(end).Value = val;
            end
        end
        if isempty(vars1)
            error('First .mat file has neither a SimulationInput object nor Simulink.Parameter variables.');
        end
    end
else
    modelName1 = erase(file1, '.slx');
    load_system(modelName1);
    mdlWks1 = get_param(modelName1, 'ModelWorkspace');
    varNames = mdlWks1.evalin('who');
    vars1 = struct('Name', {}, 'Value', {});
    for i = 1:length(varNames)
        val = mdlWks1.getVariable(varNames{i});
        if isa(val, 'Simulink.Parameter')
            vars1(end+1).Name = varNames{i};
            vars1(end).Value = val;
        end
    end
end

% --- Load File 2 ---
if endsWith(full2, '.mat')
    s2 = load(full2);
    if isfield(s2, 'in') && isa(s2.in, 'Simulink.SimulationInput')
        vars2 = s2.in.Variables;
    else
        names = fieldnames(s2);
        vars2 = struct('Name', {}, 'Value', {});
        for i = 1:length(names)
            val = s2.(names{i});
            if isa(val, 'Simulink.Parameter')
                vars2(end+1).Name = names{i};
                vars2(end).Value = val;
            end
        end
        if isempty(vars2)
            error('Second .mat file has neither a SimulationInput object nor Simulink.Parameter variables.');
        end
    end
else
    modelName2 = erase(file2, '.slx');
    load_system(modelName2);
    mdlWks2 = get_param(modelName2, 'ModelWorkspace');
    varNames = mdlWks2.evalin('who');
    vars2 = struct('Name', {}, 'Value', {});
    for i = 1:length(varNames)
        val = mdlWks2.getVariable(varNames{i});
        if isa(val, 'Simulink.Parameter')
            vars2(end+1).Name = varNames{i};
            vars2(end).Value = val;
        end
    end
end

% --- Compare ---
names1 = {vars1.Name};
names2 = {vars2.Name};

onlyIn1 = setdiff(names1, names2);
onlyIn2 = setdiff(names2, names1);
common = intersect(names1, names2);

fprintf('\nParameters only in first file:\n');
if isempty(onlyIn1)
    fprintf('  None\n');
else
    for i = 1:length(onlyIn1)
        fprintf('  %s\n', onlyIn1{i});
    end
end

fprintf('\nParameters only in second file:\n');
if isempty(onlyIn2)
    fprintf('  None\n');
else
    for i = 1:length(onlyIn2)
        fprintf('  %s\n', onlyIn2{i});
    end
end

fprintf('\nParameters with different values:\n');
diffNames = {};
for i = 1:length(common)
    name = common{i};
    v1 = vars1(strcmp(name, names1)).Value.Value;
    v2 = vars2(strcmp(name, names2)).Value.Value;

    isDifferent = false;
    if isnumeric(v1) && isnumeric(v2)
        isDifferent = any(abs(v1 - v2) > tol);
    else
        isDifferent = ~isequaln(v1, v2);
    end

    if isDifferent
        diffNames{end+1} = name; %#ok<AGROW>
        fprintf('  %s:\n    File 1: %s\n    File 2: %s\n', name, mat2str(v1), mat2str(v2));
    end
end
if isempty(diffNames)
    fprintf('  None\n');
end

% --- Optional: Save report ---
choice = questdlg('Would you like to save the comparison results to a text file?', 'Save Report', 'Yes', 'No', 'No');
if strcmp(choice, 'Yes')
    [reportFile, reportPath] = uiputfile('*.txt', 'Save Comparison Report As');
    if ~isequal(reportFile, 0)
        fid = fopen(fullfile(reportPath, reportFile), 'w');

        fprintf(fid, 'Parameters only in first file:\n');
        if isempty(onlyIn1)
            fprintf(fid, '  None\n');
        else
            for i = 1:length(onlyIn1)
                fprintf(fid, '  %s\n', onlyIn1{i});
            end
        end

        fprintf(fid, '\nParameters only in second file:\n');
        if isempty(onlyIn2)
            fprintf(fid, '  None\n');
        else
            for i = 1:length(onlyIn2)
                fprintf(fid, '  %s\n', onlyIn2{i});
            end
        end

        fprintf(fid, '\nParameters with different values:\n');
        if isempty(diffNames)
            fprintf(fid, '  None\n');
        else
            for i = 1:length(diffNames)
                name = diffNames{i};
                v1 = vars1(strcmp(name, names1)).Value.Value;
                v2 = vars2(strcmp(name, names2)).Value.Value;
                fprintf(fid, '  %s:\n    File 1: %s\n    File 2: %s\n', name, mat2str(v1), mat2str(v2));
            end
        end

        fclose(fid);
        fprintf('\nComparison report saved to: %s\n', fullfile(reportPath, reportFile));
    end
end

% --- Cleanup (clear only script-created variables) ---
clear file1 file2 path1 path2 full1 full2 i k name ...
    v1 v2 s1 s2 modelName1 modelName2 mdlWks1 mdlWks2 ...
    varNames names1 names2 vars1 vars2 onlyIn1 onlyIn2 ...
    common diffNames reportFile reportPath fid choice tol...
    isDifferent names val;
