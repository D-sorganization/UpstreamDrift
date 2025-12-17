%%% START FILE: MASTER_SCRIPT_ZTCF_ZVCF_PLOT_GENERATOR_3D.m %%%
% ZTCF and ZVCF Data Generation Script
% This script runs the main simulation (BaseData), generates Zero Torque
% Counterfactual (ZTCF) data by simulating with zeroed torques at different
% time points, calculates the Delta (Base - ZTCF), generates Zero Velocity
% Counterfactual (ZVCF) data, saves the resulting tables, and calls
% subsequent scripts for further calculations and plotting.

% The ZTCF represents the passive effect of non-torque forces.
% It is calculated by simulating with joint torques set to zero after a
% step time, matching the original position/velocity at that time point.

% The ZVCF represents the effect of joint torques in a static pose.
% It is calculated by simulating the ZVCF model with zero initial velocity
% and constant joint torques taken from the BaseData at specific time points.

% --- Initial Setup ---

% Define the root directory of your 3DModel project
% Assuming this script is located in the '3DModel' folder
projectRoot = fileparts(mfilename('fullpath'));
cd(projectRoot); % Change to the project root directory

% Add necessary script folders to the MATLAB path
addpath(fullfile(projectRoot, 'Scripts'));
addpath(fullfile(projectRoot, 'Scripts', '_BaseData Scripts'));
addpath(fullfile(projectRoot, 'Scripts', '_ZTCF Scripts'));
addpath(fullfile(projectRoot, 'Scripts', '_Delta Scripts'));
addpath(fullfile(projectRoot, 'Scripts', '_Comparison Scripts'));
addpath(fullfile(projectRoot, 'Scripts', '_ZVCF Scripts'));
% Add other necessary folders if you have them (e.g., models folder)
% addpath(fullfile(projectRoot, 'Models'));

% Define the target uniform time step for the 'Q' tables (for plotting/analysis)
TsQ = 0.0025; % Your desired uniform time step

% Load the main kinetic model and its model workspace
kineticModelName = 'GolfSwing3D_KineticallyDriven';
open_system(kineticModelName); % Open the model without displaying the window
mdlWks = get_param(kineticModelName, 'ModelWorkspace');

% Load model inputs into the model workspace (assuming ModelInputs.mat is in project root)
mdlWks.DataSource = 'MAT-File';
mdlWks.FileName = fullfile(projectRoot, 'ModelInputs.mat');
mdlWks.reload;

% Configure simulation parameters for the main run and ZTCF runs
% Set the killswitch time to a value that won't trigger during the initial Base run
assignin(mdlWks, 'KillswitchStepTime', Simulink.Parameter(1));
% Set the stop time for the main BaseData simulation
assignin(mdlWks, 'StopTime', Simulink.Parameter(0.28)); % As seen in original script

% Set up the model to return a single matlab object as output.
set_param(kineticModelName, 'ReturnWorkspaceOutputs', 'on');
set_param(kineticModelName, 'FastRestart', 'on'); % Keep Fast Restart for ZTCF loop efficiency
set_param(kineticModelName, 'MaxStep', '0.001'); % Set maximum time step

% Turn off specific warnings that are expected and deemed benign
warning('off', 'MATLAB:MKDIR:DirectoryExists'); % For mkdir
warning('off', 'Simulink:Masking:NonTunableParameterChangedDuringSimulation');
warning('off', 'Simulink:Engine:NonTunableVarChangedInFastRestart');
warning('off', 'Simulink:Engine:NonTunableVarChangedMaxWarnings');
% Consider adding more specific warnings if they appear frequently and are understood

% --- Generate BaseData ---
fprintf('Running BaseData simulation...\n');
out_base = sim(kineticModelName);

% Generate BaseData table from simulation output using refactored function
% Assuming generateDataTable3D function is available on the path
BaseData = generateDataTable3D(out_base);
fprintf('BaseData table generated.\n');

% --- Generate ZTCF Data by Looping ---
% The ZTCF simulations are run by stepping the KillswitchStepTime.
% The data point of interest from each run is where the killswitch state becomes zero.

% Define the times at which to step the killswitch for ZTCF
% Using 0:28 with a scaling factor of 1/100 as seen in your script
ztcfSampleTimes = (0:28) / 100;
numZTCSims = length(ztcfSampleTimes);

% Initialize ZTCFTable with the structure of BaseData, but empty
% This is less efficient than pre-allocation if the number of variables is large
% A better approach is to pre-allocate the table size and variable types
% Let's stick closer to your original logic for now, but pre-allocation is recommended.
ZTCFTable = BaseData(false, :); % Create an empty table with same columns as BaseData

fprintf('Generating ZTCF data (%d simulations)...\n', numZTCSims);
% Loop through each desired killswitch step time
for i = 1:numZTCSims
    currentTime = ztcfSampleTimes(i);

    % Display percentage complete
    ztcfPercentComplete = (i / numZTCSims) * 100;
    fprintf('ZTCF Progress: %.1f%%\n', ztcfPercentComplete);

    % Write step time to model workspace using setVariable on a SimulationInput object
    % Create a SimulationInput object for this specific run
    in_ztcf = Simulink.SimulationInput(kineticModelName);
    in_ztcf = in_ztcf.setVariable('KillswitchStepTime', currentTime);
    % Inherit other model workspace variables from the base model setup

    % Run the simulation for this step time
    out_ztcf = sim(in_ztcf);

    % Generate table from simulation output using refactored function
    ztcfSimData = generateDataTable3D(out_ztcf);

    % Find the row where the KillswitchState first becomes zero
    % Assuming KillswitchState is a logged signal in ztcfSimData
    row = find(ztcfSimData.KillswitchState == 0, 1);

    % Extract the data row at the killswitch time
    if ~isempty(row)
        ztcfDataRow = ztcfSimData(row, :);
        % Overwrite the time in the extracted row with the actual sample time
        ztcfDataRow.Time = currentTime;
    else
        % Handle cases where the killswitch might not reach zero (e.g., simulation ends first)
        % You might want to log a warning or skip this time point
        warning('KillswitchState did not reach zero for time %g. Skipping data point.', currentTime);
        continue; % Skip to the next iteration
    end

    % Append the extracted row to the ZTCFTable
    % This is inefficient. Pre-allocation is strongly recommended.
    ZTCFTable = [ZTCFTable; ztcfDataRow];
end
fprintf('ZTCF data generation complete.\n');

% --- Data Processing: Generate BASEQ, ZTCFQ, DELTAQ on Uniform TsQ Grid ---
% This section replaces the original re-timing logic and the call to SCRIPT_QTableTimeChange_3D.m

fprintf('Processing data and resampling to uniform TsQ grid...\n');

% Convert BaseData and ZTCF to timetables using their original time steps
BaseDataTimetable = table2timetable(BaseData, 'RowTimes', 'Time');
ZTCFTimetable = table2timetable(ZTCFTable, 'RowTimes', 'Time'); % Use ZTCFTable generated in loop

% Resample both BaseDataTimetable and ZTCFTimetable directly to the final uniform TsQ grid
% This generates the final BASEQ and ZTCFQ timetables
fprintf('Resampling BaseData and ZTCF to TsQ = %g...\n', TsQ);
BASEQTimetable = retime(BaseDataTimetable, 'regular', 'spline', 'TimeStep', seconds(TsQ));
ZTCFQTimetable = retime(ZTCFTimetable, 'regular', 'spline', 'TimeStep', seconds(TsQ));

% Calculate DELTAQ by subtracting the two re-timed timetables
% This generates the final DELTAQ timetable directly
fprintf('Calculating DELTAQ = BASEQ - ZTCFQ...\n');
DELTAQTimetable = BASEQTimetable - ZTCFQTimetable;

% Convert the uniform timetables back to tables
fprintf('Converting uniform timetables back to tables...\n');
BASEQ = timetable2table(BASEQTimetable, 'ConvertRowTimes', true);
BASEQ = renamevars(BASEQ, 't', 'Time'); % Rename the time column added by timetable2table

ZTCFQ = timetable2table(ZTCFQTimetable, 'ConvertRowTimes', true);
ZTCFQ = renamevars(ZTCFQ, 't', 'Time'); % Rename the time column

DELTAQ = timetable2table(DELTAQTimetable, 'ConvertRowTimes', true);
DELTAQ = renamevars(DELTAQ, 't', 'Time'); % Rename the time column

fprintf('BASEQ, ZTCFQ, DELTAQ tables generated on uniform TsQ grid.\n');

% --- Cleanup intermediate variables from ZTCF generation and re-timing ---
% Be careful not to clear variables needed later (like BASEQ, ZTCFQ, DELTAQ)
clear out_base out_ztcf ztcfSimData row ztcfDataRow ztcfPercentComplete;
clear BaseDataTimetable ZTCFTimetable BASEQTimetable ZTCFQTimetable DELTAQTimetable;
% You might choose to clear BaseData and ZTCFTable here if they are no longer needed
% clear BaseData ZTCFTable;


% --- Perform further Calculations using the Q Tables ---
fprintf('Performing additional calculations...\n');

% Run the correction program for linear work and linear impulse for ZTCF and DELTA.
% Assuming SCRIPT_UpdateCalcsforImpulseandWork_3D is a function now
% It should accept ZTCF and DELTA tables and return updated tables
% [ZTCF, DELTA] = SCRIPT_UpdateCalcsforImpulseandWork_3D(ZTCF, DELTA); % If using non-Q tables
[ZTCFQ, DELTAQ] = SCRIPT_UpdateCalcsforImpulseandWork_3D(ZTCFQ, DELTAQ); % Assuming it operates on Q tables

% Run the Q spacing program for the plots: (This script is now redundant and should be removed)
% cd(fullfile(projectRoot, 'Scripts')); % Already on path, no need to cd
% SCRIPT_QTableTimeChange_3D; % REMOVE THIS CALL

% Run the Calculation for Total Work and Power at Each Joint
% Assuming SCRIPT_TotalWorkandPowerCalculation_3D is a function now
% It should accept tables and return updated tables
% [BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ] = SCRIPT_TotalWorkandPowerCalculation_3D(BASE, ZTCF, DELTA, BASEQ, ZTCFQ, DELTAQ); % Example if it updates all
[BASEQ, ZTCFQ, DELTAQ] = SCRIPT_TotalWorkandPowerCalculation_3D(BASEQ, ZTCFQ, DELTAQ); % Assuming it operates on Q tables

% Generate Club and Hand Path Vectors in the Tables
% Assuming SCRIPT_CHPandMPPCalculation_3D is a function now named calculatePathVectors3D
% It should accept tables and return updated tables
[BASEQ, ZTCFQ, DELTAQ] = calculatePathVectors3D(BASEQ, ZTCFQ, DELTAQ);


% Run Table of Values Script and Generate Data for Shaft Quivers at Times of interest
% Assuming SCRIPT_TableofValues_3D is a function now named generateSummaryTableAndQuiverData3D
% It should accept BASEQ, ZTCFQ, DELTAQ and return summary tables/structs
[SummaryTable, ClubQuiverAlphaReversal, ClubQuiverMaxCHS, ClubQuiverZTCFAlphaReversal, ClubQuiverDELTAAlphaReversal] = generateSummaryTableAndQuiverData3D(BASEQ, ZTCFQ, DELTAQ);


% --- Save the tables ---
fprintf('Saving tables...\n');
tablesDir = fullfile(projectRoot, 'Tables');
mkdir(tablesDir); % Create directory if it doesn't exist

% Save the primary Q tables
save(fullfile(tablesDir, 'BASEQ.mat'), 'BASEQ');
save(fullfile(tablesDir, 'ZTCFQ.mat'), 'ZTCFQ');
save(fullfile(tablesDir, 'DELTAQ.mat'), 'DELTAQ');

% Save other calculated summary tables/structs
save(fullfile(tablesDir, 'ClubQuiverAlphaReversal.mat'), 'ClubQuiverAlphaReversal');
save(fullfile(tablesDir, 'ClubQuiverMaxCHS.mat'), 'ClubQuiverMaxCHS');
save(fullfile(tablesDir, 'ClubQuiverZTCFAlphaReversal.mat'), 'ClubQuiverZTCFAlphaReversal');
save(fullfile(tablesDir, 'ClubQuiverDELTAAlphaReversal.mat'), 'ClubQuiverDELTAAlphaReversal');
save(fullfile(tablesDir, 'SummaryTable.mat'), 'SummaryTable');

fprintf('Tables saved to %s.\n', tablesDir);

% --- Generate ZVCF Data ---
% The ZVCF simulations are run with zero initial velocity and constant
% joint torques taken from the BaseData at specific time points.

fprintf('Generating ZVCF data...\n');

% Define the ZVCF model name
zvcfModelName = 'GolfSwing3D_ZVCF';

% Load the ZVCF model and its model workspace
open_system(zvcfModelName); % Open the model
mdlWks_zvcf = get_param(zvcfModelName, 'ModelWorkspace');

% Load model inputs into the ZVCF model workspace (assuming ModelInputsZVCF.mat exists)
% Your original script copied/renamed ModelInputs.mat for this.
% A cleaner approach is to create ModelInputsZVCF.mat separately or
% configure the ZVCF model workspace differently.
% Assuming ModelInputsZVCF.mat is available for the ZVCF model's base parameters
% mdlWks_zvcf.DataSource = 'MAT-File';
% mdlWks_zvcf.FileName = fullfile(projectRoot, 'ModelInputsZVCF.mat'); % Adjust path if needed
% mdlWks_zvcf.reload;

% Set up ZVCF model simulation parameters
set_param(zvcfModelName, 'ReturnWorkspaceOutputs', 'on');
set_param(zvcfModelName, 'FastRestart', 'off'); % Keep Fast Restart off as in original ZVCF script
set_param(zvcfModelName, 'MaxStep', '0.001');
% Set the stop time for the short ZVCF simulation runs
set_param(zvcfModelName, 'StopTime', '0.05'); % As in original ZVCF script

% Define the times at which to sample BaseData for ZVCF inputs
% Using the same sample times as ZTCF for consistency
zvcfSampleTimes = ztcfSampleTimes; % Use the same times as ZTCF
numZVCFSims = length(zvcfSampleTimes);

% Initialize ZVCFTable to store results (pre-allocate is better)
% Use the structure of a table generated from a single ZVCF sim output
% Run one dummy sim or inspect the model output structure to get var names/types
fprintf('Running one dummy ZVCF simulation to get table structure...\n');
in_zvcf_dummy = Simulink.SimulationInput(zvcfModelName);
% Set minimal required variables for dummy run if model needs them
% e.g., in_zvcf_dummy = in_zvcf_dummy.setVariable('HipStartPosition', 0);
out_zvcf_dummy = sim(in_zvcf_dummy);
dummyZVCFDataTable = generateDataTable3D(out_zvcf_dummy); % Use your table generation function
ZVCFTable = table('Size', [numZVCFSims, width(dummyZVCFDataTable)], ...
                  'VariableTypes', varfun(@class, dummyZVCFDataTable(1,:), 'OutputFormat', 'cell'), ...
                  'VariableNames', dummyZVCFDataTable.Properties.VariableNames);
clear out_zvcf_dummy dummyZVCFDataTable; % Clean up dummy run variables

fprintf('Generating ZVCF data (%d simulations)...\n', numZVCFSims);

% Begin Generation of ZVCF Data by Looping
% Create an array of SimulationInput objects for ZVCF runs
in_zvcf_array = Simulink.SimulationInput.empty(0, numZVCFSims);

for i = 1:numZVCFSims
    currentTime = zvcfSampleTimes(i);

    % Display percentage complete
    zvcfPercentComplete = (i / numZVCFSims) * 100;
    fprintf('ZVCF Progress: %.1f%%\n', zvcfPercentComplete);

    % Read the joint torque and position values from BaseData at the current time
    % These values will be used as constant inputs and initial positions for the ZVCF model
    % NOTE: Ensure these column names match your 3D model's logged outputs
    hipJointTorque = interp1(BaseData.Time, BaseData.HipTorqueZInput, currentTime, 'linear'); % Assuming Z is primary input axis
    torsoJointTorque = interp1(BaseData.Time, BaseData.TorsoTorqueInput, currentTime, 'linear'); % Assuming single input
    lscapJointTorqueX = interp1(BaseData.Time, BaseData.LScapTorqueXInput, currentTime, 'linear');
    lscapJointTorqueY = interp1(BaseData.Time, BaseData.LScapTorqueYInput, currentTime, 'linear');
    rscapJointTorqueX = interp1(BaseData.Time, BaseData.RScapTorqueXInput, currentTime, 'linear');
    rscapJointTorqueY = interp1(BaseData.Time, BaseData.RScapTorqueYInput, currentTime, 'linear');
    lshoulderJointTorqueX = interp1(BaseData.Time, BaseData.LSTorqueXInput, currentTime, 'linear');
    lshoulderJointTorqueY = interp1(BaseData.Time, BaseData.LSTorqueYInput, currentTime, 'linear');
    lshoulderJointTorqueZ = interp1(BaseData.Time, BaseData.LSTorqueZInput, currentTime, 'linear');
    rshoulderJointTorqueX = interp1(BaseData.Time, BaseData.RSTorqueXInput, currentTime, 'linear');
    rshoulderJointTorqueY = interp1(BaseData.Time, BaseData.RSTorqueYInput, currentTime, 'linear');
    rshoulderJointTorqueZ = interp1(BaseData.Time, BaseData.RSTorqueZInput, currentTime, 'linear');
    lelbowJointTorque = interp1(BaseData.Time, BaseData.LETorqueInput, currentTime, 'linear'); % Assuming single input
    relbowJointTorque = interp1(BaseData.Time, BaseData.RETorqueInput, currentTime, 'linear'); % Assuming single input
    lwristJointTorqueX = interp1(BaseData.Time, BaseData.LWTorqueXInput, currentTime, 'linear');
    lwristJointTorqueY = interp1(BaseData.Time, BaseData.LWTorqueYInput, currentTime, 'linear');
    rwristJointTorqueX = interp1(BaseData.Time, BaseData.RWTorqueXInput, currentTime, 'linear');
    rwristJointTorqueY = interp1(BaseData.Time, BaseData.RWTorqueYInput, currentTime, 'linear');


    % Read the position values from BaseData at the current time and convert to degrees if necessary
    % NOTE: Ensure these column names match your 3D model's logged outputs
    % NOTE: Verify units (radians vs. degrees) expected by your ZVCF model parameters
    hipPosition = interp1(BaseData.Time, BaseData.HipPositionZ, currentTime, 'linear'); % Assuming Z position is relevant
    torsoPosition = interp1(BaseData.Time, BaseData.TorsoPosition, currentTime, 'linear'); % Assuming single position
    lscapPositionX = interp1(BaseData.Time, BaseData.LScapPositionX, currentTime, 'linear');
    lscapPositionY = interp1(BaseData.Time, BaseData.LScapPositionY, currentTime, 'linear');
    rscapPositionX = interp1(BaseData.Time, BaseData.RScapPositionX, currentTime, 'linear');
    rscapPositionY = interp1(BaseData.Time, BaseData.RScapPositionY, currentTime, 'linear');
    lshoulderPositionX = interp1(BaseData.Time, BaseData.LSPositionX, currentTime, 'linear');
    lshoulderPositionY = interp1(BaseData.Time, BaseData.LSPositionY, currentTime, 'linear');
    lshoulderPositionZ = interp1(BaseData.Time, BaseData.LSPositionZ, currentTime, 'linear');
    rshoulderPositionX = interp1(BaseData.Time, BaseData.RSPositionX, currentTime, 'linear');
    rshoulderPositionY = interp1(BaseData.Time, BaseData.RSPositionY, currentTime, 'linear');
    rshoulderPositionZ = interp1(BaseData.Time, BaseData.RSPositionZ, currentTime, 'linear');
    lelbowPosition = interp1(BaseData.Time, BaseData.LEPosition, currentTime, 'linear'); % Assuming single position
    relbowPosition = interp1(BaseData.Time, BaseData.REPosition, currentTime, 'linear'); % Assuming single position
    lwristPositionX = interp1(BaseData.Time, BaseData.LWPositionX, currentTime, 'linear');
    lwristPositionY = interp1(BaseData.Time, BaseData.LWPositionY, currentTime, 'linear');
    rwristPositionX = interp1(BaseData.Time, BaseData.RWPositionX, currentTime, 'linear');
    rwristPositionY = interp1(BaseData.Time, BaseData.RWPositionY, currentTime, 'linear');


    % Create a new SimulationInput object for this ZVCF run
    in_zvcf_array(i) = Simulink.SimulationInput(zvcfModelName);

    % Assign in the torque values as constant inputs to the ZVCF model using setVariable
    % NOTE: Ensure parameter names match your ZVCF model's input blocks/parameters
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueHip', hipJointTorque);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueTorso', torsoJointTorque);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueLScapX', lscapJointTorqueX);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueLScapY', lscapJointTorqueY);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueRScapX', rscapJointTorqueX);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueRScapY', rscapJointTorqueY);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueLShoulderX', lshoulderJointTorqueX);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueLShoulderY', lshoulderJointTorqueY);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueLShoulderZ', lshoulderJointTorqueZ);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueRShoulderX', rshoulderJointTorqueX);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueRShoulderY', rshoulderJointTorqueY);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueRShoulderZ', rshoulderJointTorqueZ);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueLElbow', lelbowJointTorque);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueRElbow', relbowJointTorque);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueLWristX', lwristJointTorqueX);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueLWristY', lwristJointTorqueY);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueRWristX', rwristJointTorqueX);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('JointTorqueRWristY', rwristJointTorqueY);


    % Assign in position values as initial positions for the ZVCF model using setVariable
    % Convert to degrees if the ZVCF model parameters expect degrees
    % NOTE: Ensure parameter names match your ZVCF model's initial condition parameters
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('HipStartPosition', hipPosition * 180/pi); % Example conversion
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('TorsoStartPosition', torsoPosition * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LScapStartPositionX', lscapPositionX * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LScapStartPositionY', lscapPositionY * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RScapStartPositionX', rscapPositionX * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RScapStartPositionY', rscapPositionY * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LShoulderStartPositionX', lshoulderPositionX * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LShoulderStartPositionY', lshoulderPositionY * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LShoulderStartPositionZ', lshoulderPositionZ * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RShoulderStartPositionX', rshoulderPositionX * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RShoulderStartPositionY', rshoulderPositionY * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RShoulderStartPositionZ', rshoulderPositionZ * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LElbowStartPosition', lelbowPosition * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RElbowStartPosition', relbowPosition * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LWristStartPositionX', lwristPositionX * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LWristStartPositionY', lwristPositionY * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RWristStartPositionX', rwristPositionX * 180/pi);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RWristStartPositionY', rwristPositionY * 180/pi);


    % Set initial velocities to zero using setVariable
    % NOTE: Ensure parameter names match your ZVCF model's initial condition parameters
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('HipStartVelocity', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('TorsoStartVelocity', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LScapStartVelocityX', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LScapStartVelocityY', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RScapStartVelocityX', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RScapStartVelocityY', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LShoulderStartVelocityX', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LShoulderStartVelocityY', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LShoulderStartVelocityZ', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RShoulderStartVelocityX', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RShoulderStartVelocityY', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RShoulderStartVelocityZ', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LElbowStartVelocity', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RElbowStartVelocity', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LWristStartVelocityX', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('LWristStartVelocityY', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RWristStartVelocityX', 0);
    in_zvcf_array(i) = in_zvcf_array(i).setVariable('RWristStartVelocityY', 0);


    % Ensure logging is configured in the model or via the input object
    % in_zvcf_array(i) = in_zvcf_array(i).setLoggingOption('all'); % Example if needed
end

% Run all ZVCF simulations (consider using parsim if Parallel Computing Toolbox is available)
fprintf('Running ZVCF simulations...\n');
% out_zvcf_results = sim(in_zvcf_array); % Sequential
out_zvcf_results = parsim(in_zvcf_array, 'ShowProgress', 'on'); % Parallel (requires PCT)
fprintf('ZVCF simulations complete.\n');

% Process ZVCF results and populate ZVCFTable
fprintf('Processing ZVCF simulation outputs...\n');
for i = 1:numZVCFSims
    currentOutput = out_zvcf_results(i);
    currentTime = zvcfSampleTimes(i); % Get the sample time for this run

    % Generate table from simulation output using refactored function
    currentZVCFDataTable = generateDataTable3D(currentOutput);

    % Extract the data row at time zero (or the specific point needed for ZVCF)
    % Your original code extracts the first row (time zero)
    zvcfRow = currentZVCFDataTable(1,:);

    % Assign the correct time from the loop (currentTime)
    zvcfRow.Time = currentTime; % Assign the original sample time

    % Populate the pre-allocated ZVCFTable
    ZVCFTable(i,:) = zvcfRow;
end
fprintf('ZVCFTable generated.\n');

% --- Save the ZVCF Tables ---
fprintf('Saving ZVCF tables...\n');
tablesDir = fullfile(projectRoot, 'Tables');
mkdir(tablesDir); % Ensure directory exists

% Assuming you also want a ZVCFTableQ on the TsQ grid
% Since ZVCFTable is already on the zvcfSampleTimes grid (which is ztcfSampleTimes)
% and ztcfSampleTimes is likely a subset of the TsQ grid,
% you might need to re-time ZVCFTable to the full TsQ grid if needed for comparison/plotting.
% Let's re-time ZVCFTable to TsQ to create ZVCFTableQ for consistency with BASEQ/ZTCFQ/DELTAQ.

ZVCFTableTimetable = table2timetable(ZVCFTable, 'RowTimes', 'Time');
ZVCFTableQTimetable = retime(ZVCFTableTimetable, 'regular', 'spline', 'TimeStep', seconds(TsQ));
ZVCFTableQ = timetable2table(ZVCFTableQTimetable, 'ConvertRowTimes', true);
ZVCFTableQ = renamevars(ZVCFTableQ, 't', 'Time');

save(fullfile(tablesDir, 'ZVCFTable.mat'), 'ZVCFTable');
save(fullfile(tablesDir, 'ZVCFTableQ.mat'), 'ZVCFTableQ');
fprintf('ZVCF tables saved to %s.\n', tablesDir);

% --- Call Plotting Scripts ---
fprintf('Generating plots...\n');
% Assuming plotting scripts are functions that accept the Q tables
% and other necessary data (like the ClubQuiver structs) as inputs.
% Update these calls to pass the tables and structs.

% Example calls (adjust function names and arguments as needed)
% MASTER_SCRIPT_BaseDataCharts_3D(BASEQ, ClubQuiverAlphaReversal, ClubQuiverMaxCHS); % Example
% MASTER_SCRIPT_ZTCFCharts_3D(ZTCFQ, ClubQuiverZTCFAlphaReversal, ClubQuiverMaxCHS); % Example
% MASTER_SCRIPT_DeltaCharts_3D(DELTAQ, ClubQuiverDELTAAlphaReversal, ClubQuiverMaxCHS); % Example
% MASTER_SCRIPT_ComparisonCharts_3D(BASEQ, ZTCFQ, DELTAQ, ClubQuiverAlphaReversal, ClubQuiverMaxCHS, ClubQuiverZTCFAlphaReversal, ClubQuiverDELTAAlphaReversal); % Example
% MASTER_SCRIPT_ZVCF_CHARTS_3D(ZVCFTableQ, DELTAQ); % Example ZVCF vs Delta comparison

% Your original script just called other master scripts.
% If those master scripts handle loading data themselves, ensure they load the correct .mat files saved above.
% If they are also converted to functions, call them with the generated tables.
% For now, let's assume they load from the 'Tables' directory.
cd(fullfile(projectRoot, 'Scripts')); % Change back to Scripts directory to call other scripts by name

% Call the main plotting orchestrator script/function
% Assuming MASTER_SCRIPT_AllCharts_3D is now a function
MASTER_SCRIPT_AllCharts_3D(BASEQ, ZTCFQ, DELTAQ, ZVCFTableQ); % Pass the Q tables and ZVCFQ

fprintf('Plot generation complete.\n');

% --- Generate Results Folder ---
fprintf('Generating results folder...\n');
% Assuming SCRIPT_ResultsFolderGeneration_3D is updated to use fullfile
% and copy the correct files (including the new Q tables)
SCRIPT_ResultsFolderGeneration_3D;
fprintf('Results folder generated.\n');

% --- Final Cleanup ---
% Close models if desired
close_system(kineticModelName, 0); % Close without saving changes
close_system(zvcfModelName, 0); % Close without saving changes

% Clear variables that are no longer needed
clear mdlWks mdlWks_zvcf;
clear BaseData ZTCFTable; % Clear the original time step tables
clear BASEQTimetable ZTCFQTimetable DELTAQTimetable ZVCFTableTimetable ZVCFTableQTimetable;
clear zvcfSampleTimes numZVCFSims in_zvcf_array zvcfPercentComplete currentOutput currentZVCFDataTable zvcfRow;
clear hipJointTorque torsoJointTorque lscapJointTorqueX lscapJointTorqueY rscapJointTorqueX rscapJointTorqueY lshoulderJointTorqueX lshoulderJointTorqueY lshoulderJointTorqueZ rshoulderJointTorqueX rshoulderJointTorqueY rshoulderJointTorqueZ lelbowJointTorque relbowJointTorque lwristJointTorqueX lwristJointTorqueY rwristJointTorqueX rwristJointTorqueY;
clear hipPosition torsoPosition lscapPositionX lscapPositionY rscapPositionX rscapPositionY lshoulderPositionX lshoulderPositionY lshoulderPositionZ rshoulderPositionX rshoulderPositionY rshoulderPositionZ lelbowPosition relbowPosition lwristPositionX lwristPositionY rwristPositionX rwristPositionY;
clear in_ztcf_array in_ztcf currentTime numZTCSims ztcfSampleTimes;
clear projectRoot tablesDir; % Clear path variables if not needed
clear SummaryTable ClubQuiverAlphaReversal ClubQuiverMaxCHS ClubQuiverZTCFAlphaReversal ClubQuiverDELTAAlphaReversal; % Clear these after saving if not needed later

fprintf('Master script execution finished.\n');

%%% END FILE: MASTER_SCRIPT_ZTCF_ZVCF_PLOT_GENERATOR_3D.m %%%
