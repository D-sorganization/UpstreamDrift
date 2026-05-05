function simOut = run_ml_polynomial_input_swing(polynomialInputFile, modelName)
%RUN_ML_POLYNOMIAL_INPUT_SWING Run GolfSwing3D_Kinetic with ML polynomial inputs.
%
% simOut = run_ml_polynomial_input_swing(polynomialInputFile)
% loads a MAT file produced by export_torque_polynomials.py and applies its
% scalar coefficient variables to a Simulink.SimulationInput object.

if nargin < 1 || isempty(polynomialInputFile)
    thisFile = mfilename('fullpath');
    mlDir = fileparts(fileparts(thisFile));
    polynomialInputFile = fullfile( ...
        mlDir, 'data', 'processed', 'ml_torque_polynomial_inputs.mat');
end

if nargin < 2 || isempty(modelName)
    modelName = 'GolfSwing3D_Kinetic';
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
inputData = load(polynomialInputFile);
names = fieldnames(inputData);

for idx = 1:numel(names)
    name = names{idx};
    value = inputData.(name);
    if isnumeric(value) && isscalar(value)
        simIn = simIn.setVariable(name, double(value));
    end
end

simOut = sim(simIn);
end
