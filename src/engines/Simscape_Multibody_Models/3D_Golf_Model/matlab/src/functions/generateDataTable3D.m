function Data = generateDataTable3D(simOutput)
% GENERATEDATATABLE3D Extracts data from 3D Simulink simulation output into a table.
%   Data = GENERATEDATATABLE3D(simOutput) processes the Simulink
%   simulation output object 'simOutput' from the 3D golf swing model
%   and extracts the time vector and logged signals into a MATLAB table.
%   It also calculates and adds 3D segment direction vectors (Grip, Shaft,
%   Forearm, Upper Arm, Shoulder) to the table for visualization or analysis.
%
%   Input:
%       simOutput - A Simulink simulation output object logged as a single
%                   object, expected to contain 'tout' (time) and 'logsout'
%                   (logged signals). 'logsout' should be a Dataset object.
%
%   Output:
%       Data - A MATLAB table containing simulation time and all logged
%              signals from 'simOutput.logsout', plus calculated 3D segment vectors.

% --- Extract Time and Logged Signals ---

% Generate a table with a time column from the simulation output time vector.
Time = simOutput.tout;
Data = table(Time, 'VariableNames', {'Time'});

% Loop through each dataset element in logsout to add it to the table.
% This method is robust to changes in the number or order of logged signals
% in the Simulink model, as long as the signal names are valid table variable names.
if isfield(simOutput, 'logsout') && ~isempty(simOutput.logsout)
    for i = 1:simOutput.logsout.numElements
        % Get signal element from the logsout Dataset
        signalElement = simOutput.logsout.getElement(i);

        % Get signal name and data
        signalName = signalElement.Name;
        signalData = signalElement.Values.Data;

        % Add the data as a new column in the table, using the signal name.
        % The .(signalName) syntax allows using the string signalName as a field/variable name.
        % Ensure signalName is a valid MATLAB table variable name if necessary,
        % though logsout names are usually compatible.
        Data.(signalName) = signalData;
    end
else
    warning('Simulink output object does not contain logsout data.');
    % The table will contain only the time vector in this case.
end


% --- Generate 3D Segment Vector Components ---
% These vectors are calculated for visualization (e.g., quiver plots) or
% further analysis based on the positions of key points logged from the model.

% Grip Scale Factor (Optional: Scale up grip vector for graphics if needed)
% Adjust this value based on desired visualization size relative to other vectors.
GripScale = 1.5;

% Generate Grip Vector in Table (e.g., from Butt to RW, scaled)
% Assumes 'RWx', 'RWy', 'RWz', 'Buttx', 'Butty', 'Buttz' are logged signals.
if all(ismember({'RWx', 'RWy', 'RWz', 'Buttx', 'Butty', 'Buttz'}, Data.Properties.VariableNames))
    Data.Gripdx = GripScale .* (Data.RWx - Data.Buttx);
    Data.Gripdy = GripScale .* (Data.RWy - Data.Butty);
    Data.Gripdz = GripScale .* (Data.RWz - Data.Buttz);
end
% GripScale variable is local to this function, no need to clear explicitly.

% Generate Shaft Vector in Table (e.g., from RW to CH)
% Assumes 'CHx', 'CHy', 'CHz', 'RWx', 'RWy', 'RWz' are logged signals.
if all(ismember({'CHx', 'CHy', 'CHz', 'RWx', 'RWy', 'RWz'}, Data.Properties.VariableNames))
    Data.Shaftdx = Data.CHx - Data.RWx;
    Data.Shaftdy = Data.CHy - Data.RWy;
    Data.Shaftdz = Data.CHz - Data.RWz;
end

% Generate Left Forearm Vector in Table (e.g., from LE to LW)
% Assumes 'LWx', 'LWy', 'LWz', 'LEx', 'LEy', 'LEz' are logged signals.
if all(ismember({'LWx', 'LWy', 'LWz', 'LEx', 'LEy', 'LEz'}, Data.Properties.VariableNames))
    Data.LeftForearmdx = Data.LWx - Data.LEx;
    Data.LeftForearmdy = Data.LWy - Data.LEy;
    Data.LeftForearmdz = Data.LWz - Data.LEz;
end

% Generate Right Forearm Vector in Table (e.g., from RE to RW)
% Assumes 'RWx', 'RWy', 'RWz', 'REx', 'REy', 'REz' are logged signals.
if all(ismember({'RWx', 'RWy', 'RWz', 'REx', 'REy', 'REz'}, Data.Properties.VariableNames))
    Data.RightForearmdx = Data.RWx - Data.REx;
    Data.RightForearmdy = Data.RWy - Data.REy;
    Data.RightForearmdz = Data.RWz - Data.REz;
end

% Generate Left Upper Arm Vector in Table (e.g., from LS to LE)
% Assumes 'LEx', 'LEy', 'LEz', 'LSx', 'LSy', 'LSz' are logged signals.
if all(ismember({'LEx', 'LEy', 'LEz', 'LSx', 'LSy', 'LSz'}, Data.Properties.VariableNames))
    Data.LeftArmdx = Data.LEx - Data.LSx;
    Data.LeftArmdy = Data.LEy - Data.LSy;
    Data.LeftArmdz = Data.LEz - Data.LSz;
end

% Generate Right Upper Arm Vector in Table (e.g., from RS to RE)
% Assumes 'REx', 'REy', 'REz', 'RSx', 'RSy', 'RSz' are logged signals.
if all(ismember({'REx', 'REy', 'REz', 'RSx', 'RSy', 'RSz'}, Data.Properties.VariableNames))
    Data.RightArmdx = Data.REx - Data.RSx;
    Data.RightArmdy = Data.REy - Data.RSy;
    Data.RightArmdz = Data.REz - Data.RSz;
end

% Generate Left Shoulder Vector (e.g., from HUB to LS)
% Assumes 'LSx', 'LSy', 'LSz', 'HUBx', 'HUBy', 'HUBz' are logged signals.
if all(ismember({'LSx', 'LSy', 'LSz', 'HUBx', 'HUBy', 'HUBz'}, Data.Properties.VariableNames))
    Data.LeftShoulderdx = Data.LSx - Data.HUBx;
    Data.LeftShoulderdy = Data.LSy - Data.HUBy;
    Data.LeftShoulderdz = Data.LSz - Data.HUBz;
end

% Generate Right Shoulder Vector (e.g., from HUB to RS)
% Assumes 'RSx', 'RSy', 'RSz', 'HUBx', 'HUBy', 'HUBz' are logged signals.
if all(ismember({'RSx', 'RSy', 'RSz', 'HUBx', 'HUBy', 'HUBz'}, Data.Properties.VariableNames))
    Data.RightShoulderdx = Data.RSx - Data.HUBx;
    Data.RightShoulderdy = Data.RSy - Data.HUBy;
    Data.RightShoulderdz = Data.RSz - Data.HUBz;
end

% Variables created inside this function (Time, i, signalElement, signalName,
% signalData, GripScale, etc.) are local and automatically cleared when
% the function finishes execution. Explicit 'clear' statements are not needed.

end
