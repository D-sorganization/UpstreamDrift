function ml_workflow_gui()
%ML_WORKFLOW_GUI UI for the golf ML club-control workflow.

thisFile = mfilename('fullpath');
mlDir = fileparts(fileparts(thisFile));
dataDir = fullfile(mlDir, 'data', 'processed');

fig = uifigure('Name', 'Golf ML Club Control', 'Position', [100 100 1040 720]);
root = uigridlayout(fig, [3 1]);
root.RowHeight = {150, '1x', 150};
root.ColumnWidth = {'1x'};

shared = uigridlayout(root, [5 6]);
shared.Layout.Row = 1;
shared.RowHeight = {26, 26, 26, 26, 26};
shared.ColumnWidth = {130, '1x', 120, '1x', 120, '1x'};

uilabel(shared, 'Text', 'Scenario');
scenarioDropDown = uidropdown(shared, ...
    'Items', {'full-swing', 'downswing'}, ...
    'Value', 'downswing');

uilabel(shared, 'Text', 'Python');
pythonField = uieditfield(shared, 'text', 'Value', 'py -3.12');

uilabel(shared, 'Text', 'Checkpoint');
checkpointField = uieditfield(shared, 'text', ...
    'Value', fullfile(mlDir, 'runs', 'club_direct_10_cpu', 'best_model.pt'));

uilabel(shared, 'Text', 'Raw Target CSV');
rawTargetField = uieditfield(shared, 'text', ...
    'Value', fullfile(dataDir, 'TW_ProV1_club_target.csv'));

uilabel(shared, 'Text', 'Target CSV');
targetField = uieditfield(shared, 'text', ...
    'Value', fullfile(dataDir, 'TW_ProV1_club_target.csv'));

uilabel(shared, 'Text', 'Sim CSV');
simField = uieditfield(shared, 'text', ...
    'Value', fullfile(dataDir, 'simulated_club_motion.csv'));

uilabel(shared, 'Text', 'Start MAT');
startField = uieditfield(shared, 'text', ...
    'Value', fullfile(dataDir, 'ml_downswing_start_state.mat'));

uilabel(shared, 'Text', 'Torque CSV');
torqueField = uieditfield(shared, 'text', ...
    'Value', fullfile(dataDir, 'optimized_club_torques.csv'));

uilabel(shared, 'Text', 'Polynomial MAT');
polyField = uieditfield(shared, 'text', ...
    'Value', fullfile(dataDir, 'ml_torque_polynomial_inputs.mat'));

uilabel(shared, 'Text', 'Reference Body CSV');
referenceField = uieditfield(shared, 'text', ...
    'Value', fullfile(dataDir, 'reference_body_state.csv'));

uilabel(shared, 'Text', 'Joint Velocity CSV');
jointVelocityField = uieditfield(shared, 'text', 'Value', '');

uilabel(shared, 'Text', 'Calibration JSON');
calibrationJsonField = uieditfield(shared, 'text', ...
    'Value', scenarioPath('club_target_calibration.json'));

tabs = uitabgroup(root);
tabs.Layout.Row = 2;

targetTab = uitab(tabs, 'Title', 'Target + Replay');
targetGrid = uigridlayout(targetTab, [4 4]);
targetGrid.RowHeight = {32, 32, 32, '1x'};
targetGrid.ColumnWidth = {'1x', '1x', '1x', '1x'};
uibutton(targetGrid, 'Text', 'Prepare Target', 'ButtonPushedFcn', @(~, ~) runPrepareTarget());
uibutton(targetGrid, 'Text', 'Slice Target', 'ButtonPushedFcn', @(~, ~) runSliceTarget());
uibutton(targetGrid, 'Text', 'Calibrate Target', 'ButtonPushedFcn', @(~, ~) runCalibrateTarget());
uibutton(targetGrid, 'Text', 'Validate Calibration', 'ButtonPushedFcn', @(~, ~) runValidateCalibration());
uibutton(targetGrid, 'Text', 'Export Start', 'ButtonPushedFcn', @(~, ~) runExportStart());
uibutton(targetGrid, 'Text', 'Run Model', 'ButtonPushedFcn', @(~, ~) runModel());
uibutton(targetGrid, 'Text', 'Compare Motion', 'ButtonPushedFcn', @(~, ~) runCompareMotion());
uibutton(targetGrid, 'Text', 'Evaluate Match', 'ButtonPushedFcn', @(~, ~) runEvaluateMatching());

surrogateTab = uitab(tabs, 'Title', 'Surrogate Sweep');
surrogateGrid = uigridlayout(surrogateTab, [5 4]);
surrogateGrid.RowHeight = {30, 30, 30, 32, '1x'};
surrogateGrid.ColumnWidth = {150, '1x', 150, '1x'};
uilabel(surrogateGrid, 'Text', 'Effort Weights');
effortGridField = uieditfield(surrogateGrid, 'text', 'Value', '1e-8,1e-7,1e-6');
uilabel(surrogateGrid, 'Text', 'Smooth Weights');
smoothGridField = uieditfield(surrogateGrid, 'text', 'Value', '1e-10,1e-9,1e-8');
uilabel(surrogateGrid, 'Text', 'Pareto Output');
paretoDirField = uieditfield(surrogateGrid, 'text', ...
    'Value', fullfile(dataDir, 'pareto_sweep'));
uilabel(surrogateGrid, 'Text', 'Steps');
paretoStepsField = uieditfield(surrogateGrid, 'numeric', 'Value', 500, ...
    'Limits', [1 Inf], 'RoundFractionalValues', 'on');
uibutton(surrogateGrid, 'Text', 'Optimize Tau', 'ButtonPushedFcn', @(~, ~) runOptimizeTorque());
uibutton(surrogateGrid, 'Text', 'Pareto Sweep', 'ButtonPushedFcn', @(~, ~) runParetoSweep());
uibutton(surrogateGrid, 'Text', 'Export Poly', 'ButtonPushedFcn', @(~, ~) runExportPolynomial());
uibutton(surrogateGrid, 'Text', 'Run + Evaluate', 'ButtonPushedFcn', @(~, ~) runModelThenEvaluate());

frameTab = uitab(tabs, 'Title', 'Frame Search');
frameGrid = uigridlayout(frameTab, [6 4]);
frameGrid.RowHeight = {30, 30, 30, 30, 32, '1x'};
frameGrid.ColumnWidth = {160, '1x', 150, '1x'};
uilabel(frameGrid, 'Text', 'Manifest JSON');
frameManifestField = uieditfield(frameGrid, 'text', ...
    'Value', fullfile(dataDir, 'frame_by_frame_search.json'));
frameManifestField.Layout.Column = [2 4];
uilabel(frameGrid, 'Text', 'Candidate Step');
candidateStepField = uieditfield(frameGrid, 'numeric', 'Value', 5.0, 'Limits', [0 Inf]);
uilabel(frameGrid, 'Text', 'Candidate Levels');
candidateLevelsField = uieditfield(frameGrid, 'text', 'Value', '-1,0,1');
uilabel(frameGrid, 'Text', 'Use Parallel');
parallelDropDown = uidropdown(frameGrid, ...
    'Items', {'auto', 'always', 'never'}, ...
    'Value', 'auto');
uilabel(frameGrid, 'Text', 'Control Columns');
frameControlField = uieditfield(frameGrid, 'text', 'Value', '');
frameControlField.Layout.Column = [2 4];
uibutton(frameGrid, 'Text', 'Prepare Manifest', 'ButtonPushedFcn', @(~, ~) runPrepareFrameSearch());
uibutton(frameGrid, 'Text', 'Run Search', 'ButtonPushedFcn', @(~, ~) runFrameSearch());
uibutton(frameGrid, 'Text', 'Replay Result', 'ButtonPushedFcn', @(~, ~) runFrameReplay());
uibutton(frameGrid, 'Text', 'Open Notes', 'ButtonPushedFcn', @(~, ~) openFrameDocs());

diagnosticsTab = uitab(tabs, 'Title', 'Diagnostics');
diagGrid = uigridlayout(diagnosticsTab, [3 4]);
diagGrid.RowHeight = {32, 32, '1x'};
diagGrid.ColumnWidth = {'1x', '1x', '1x', '1x'};
uibutton(diagGrid, 'Text', 'Validate Calibration', 'ButtonPushedFcn', @(~, ~) runValidateCalibration());
uibutton(diagGrid, 'Text', 'Evaluate Match', 'ButtonPushedFcn', @(~, ~) runEvaluateMatching());
uibutton(diagGrid, 'Text', 'Open Reports', 'ButtonPushedFcn', @(~, ~) winopen(dataDir));
uibutton(diagGrid, 'Text', 'Open ML Folder', 'ButtonPushedFcn', @(~, ~) winopen(mlDir));

logArea = uitextarea(root, 'Editable', 'off');
logArea.Layout.Row = 3;

    function runPrepareTarget()
        command = sprintf('%s "%s" --output "%s"', ...
            pythonField.Value, fullfile(mlDir, 'prepare_club_target_trajectory.py'), rawTargetField.Value);
        runCommand(command);
        targetField.Value = rawTargetField.Value;
    end

    function runSliceTarget()
        slicedCsv = scenarioPath('club_target_sliced.csv');
        command = sprintf('%s "%s" --input-csv "%s" --output-csv "%s" --scenario %s --reset-time', ...
            pythonField.Value, fullfile(mlDir, 'slice_club_target.py'), targetField.Value, slicedCsv, scenarioDropDown.Value);
        runCommand(command);
        targetField.Value = slicedCsv;
        startField.Value = scenarioPath('start_state.mat');
    end

    function runCalibrateTarget()
        calibratedCsv = scenarioPath('club_target_calibrated.csv');
        calibrationJson = scenarioPath('club_target_calibration.json');
        command = sprintf('%s "%s" --target-csv "%s" --sim-csv "%s" --output-csv "%s" --output-json "%s"', ...
            pythonField.Value, fullfile(mlDir, 'calibrate_club_target_to_sim.py'), targetField.Value, simField.Value, calibratedCsv, calibrationJson);
        runCommand(command);
        calibrationJsonField.Value = calibrationJson;
        targetField.Value = calibratedCsv;
    end

    function runValidateCalibration()
        outputDir = scenarioPath('calibration_validation');
        command = sprintf('%s "%s" --measured-target-csv "%s" --calibrated-target-csv "%s" --sim-csv "%s" --transform-json "%s" --output-dir "%s" --run-label %s', ...
            pythonField.Value, fullfile(mlDir, 'validate_club_calibration.py'), rawTargetField.Value, targetField.Value, simField.Value, calibrationJsonField.Value, outputDir, scenarioDropDown.Value);
        runCommand(command);
    end

    function runExportStart()
        export_start_state_from_input_file(scenarioDropDown.Value, startField.Value);
        appendLog(sprintf('Exported %s', startField.Value));
    end

    function runOptimizeTorque()
        command = sprintf('%s "%s" --checkpoint "%s" --desired-club-csv "%s" --reference-body-csv "%s" --output-csv "%s"', ...
            pythonField.Value, fullfile(mlDir, 'optimize_torque_sequence_for_club.py'), checkpointField.Value, targetField.Value, referenceField.Value, torqueField.Value);
        runCommand(command);
    end

    function runParetoSweep()
        command = sprintf('%s "%s" --checkpoint "%s" --desired-club-csv "%s" --reference-body-csv "%s" --output-dir "%s" --effort-weights "%s" --smoothness-weights "%s" --steps %d --scenario %s', ...
            pythonField.Value, fullfile(mlDir, 'run_matching_pareto_sweep.py'), checkpointField.Value, targetField.Value, referenceField.Value, paretoDirField.Value, effortGridField.Value, smoothGridField.Value, round(paretoStepsField.Value), scenarioDropDown.Value);
        runCommand(command);
    end

    function runExportPolynomial()
        command = sprintf('%s "%s" --torque-csv "%s" --output-mat "%s"', ...
            pythonField.Value, fullfile(mlDir, 'export_torque_polynomials.py'), torqueField.Value, polyField.Value);
        runCommand(command);
    end

    function runModel()
        simOut = run_ml_polynomial_input_swing(polyField.Value, 'GolfSwing3D_Kinetic', startField.Value);
        assignin('base', 'mlWorkflowSimOut', simOut);
        export_simulated_club_csv(simOut, simField.Value);
        appendLog(sprintf('Ran model and exported %s', simField.Value));
    end

    function runModelThenEvaluate()
        runModel();
        runEvaluateMatching();
    end

    function runCompareMotion()
        outputJson = scenarioPath('club_motion_comparison.json');
        outputPng = scenarioPath('club_motion_comparison.png');
        command = sprintf('%s "%s" --target-csv "%s" --sim-csv "%s" --output-json "%s" --output-png "%s"', ...
            pythonField.Value, fullfile(mlDir, 'compare_simulated_club_motion.py'), targetField.Value, simField.Value, outputJson, outputPng);
        runCommand(command);
    end

    function runEvaluateMatching()
        outputDir = scenarioPath('matching_reports');
        qdotArg = '';
        if strlength(string(jointVelocityField.Value)) > 0
            qdotArg = sprintf(' --joint-velocity-csv "%s"', jointVelocityField.Value);
        end
        command = sprintf('%s "%s" --target-csv "%s" --sim-csv "%s" --torque-csv "%s"%s --output-dir "%s" --scenario %s --run-label %s', ...
            pythonField.Value, fullfile(mlDir, 'evaluate_matching_workflow.py'), targetField.Value, simField.Value, torqueField.Value, qdotArg, outputDir, scenarioDropDown.Value, scenarioDropDown.Value);
        runCommand(command);
    end

    function runPrepareFrameSearch()
        controlArg = '';
        if strlength(string(frameControlField.Value)) > 0
            controlArg = sprintf(' --control-columns "%s"', frameControlField.Value);
        end
        command = sprintf('%s "%s" --desired-target-csv "%s" --output-json "%s" --starting-state-file "%s" --torque-output-csv "%s" --polynomial-output-mat "%s" --candidate-step %.8g --candidate-levels "%s" --use-parallel %s%s', ...
            pythonField.Value, fullfile(mlDir, 'prepare_frame_by_frame_search.py'), targetField.Value, frameManifestField.Value, startField.Value, frameTorqueCsv(), framePolynomialMat(), candidateStepField.Value, candidateLevelsField.Value, parallelDropDown.Value, controlArg);
        runCommand(command);
        torqueField.Value = frameTorqueCsv();
        polyField.Value = framePolynomialMat();
    end

    function runFrameSearch()
        addpath(fullfile(mlDir, 'matlab'));
        summary = run_frame_by_frame_torque_search(frameManifestField.Value);
        assignin('base', 'mlFrameSearchSummary', summary);
        appendLog(sprintf('Frame search wrote %s', summary.smoothedTorqueCsv));
        torqueField.Value = summary.smoothedTorqueCsv;
        polyField.Value = summary.polynomialMat;
    end

    function runFrameReplay()
        torqueField.Value = frameSmoothedTorqueCsv();
        polyField.Value = framePolynomialMat();
        runModelThenEvaluate();
    end

    function openFrameDocs()
        winopen(fullfile(mlDir, 'FRAME_BY_FRAME_OPTIMIZATION.md'));
    end

    function outputPath = scenarioPath(fileName)
        outputPath = fullfile(dataDir, sprintf('%s_%s', scenarioDropDown.Value, fileName));
    end

    function outputPath = frameTorqueCsv()
        outputPath = fullfile(dataDir, sprintf('%s_frame_by_frame_torque_sequence.csv', scenarioDropDown.Value));
    end

    function outputPath = frameSmoothedTorqueCsv()
        [folder, name, ext] = fileparts(frameTorqueCsv());
        outputPath = fullfile(folder, strcat(name, '_smoothed', ext));
    end

    function outputPath = framePolynomialMat()
        outputPath = fullfile(dataDir, sprintf('%s_frame_by_frame_torque_polynomials.mat', scenarioDropDown.Value));
    end

    function runCommand(command)
        appendLog(command);
        [status, output] = system(command);
        appendLog(output);
        if status ~= 0
            appendLog(sprintf('Command failed with status %d', status));
        end
    end

    function appendLog(message)
        current = logArea.Value;
        if ischar(current)
            current = {current};
        end
        lines = splitlines(string(message));
        logArea.Value = [current; cellstr(lines(:))];
        drawnow;
    end
end
