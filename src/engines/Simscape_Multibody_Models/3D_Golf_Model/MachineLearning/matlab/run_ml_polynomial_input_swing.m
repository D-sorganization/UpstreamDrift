function simOut = run_ml_polynomial_input_swing(polynomialInputFile, modelName, startingStateFile)
%RUN_ML_POLYNOMIAL_INPUT_SWING Run GolfSwing3D_Kinetic with ML polynomial inputs.
%
% simOut = run_ml_polynomial_input_swing(polynomialInputFile)
% loads a MAT file produced by export_torque_polynomials.py and applies its
% scalar coefficient variables to a Simulink.SimulationInput object.
%
% simOut = run_ml_polynomial_input_swing(polynomialInputFile, modelName, startingStateFile)
% also applies a starting-state MAT file produced by
% export_start_state_from_input_file.m. Starting state variables are applied
% separately from the polynomial torque coefficients so address and
% top-of-backswing workflows can reuse the same torque export machinery.

if nargin < 1 || isempty(polynomialInputFile)
    thisFile = mfilename('fullpath');
    mlDir = fileparts(fileparts(thisFile));
    polynomialInputFile = fullfile( ...
        mlDir, 'data', 'processed', 'ml_torque_polynomial_inputs.mat');
end

if nargin < 2 || isempty(modelName)
    modelName = 'GolfSwing3D_Kinetic';
end

if nargin < 3
    startingStateFile = '';
end

repoModelDir = fullfile( ...
    fileparts(fileparts(mfilename('fullpath'))), '..', ...
    'matlab', 'src', 'model');

if exist(repoModelDir, 'dir')
    addpath(repoModelDir);
end

if ~isfile(polynomialInputFile)
    error('ML polynomial input file not found: %s', polynomialInputFile);
end

if ~bdIsLoaded(modelName)
    load_system(modelName);
end

simIn = Simulink.SimulationInput(modelName);

if ~isempty(startingStateFile)
    if ~isfile(startingStateFile)
        error('ML starting state file not found: %s', startingStateFile);
    end
    simIn = applyScalarVariables(simIn, startingStateFile);
end

simIn = applyScalarVariables(simIn, polynomialInputFile);
simOut = sim(simIn);
end

function simIn = applyScalarVariables(simIn, matFile)
inputData = load(matFile);
names = fieldnames(inputData);

for idx = 1:numel(names)
    name = names{idx};
    value = inputData.(name);
    if isa(value, 'Simulink.Parameter')
        value = value.Value;
    end
    if isnumeric(value) && isscalar(value)
        simIn = simIn.setVariable(name, double(value));
    end
end
end
