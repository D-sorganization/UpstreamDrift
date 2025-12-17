% exportAllLoggedRuns.m
% Automatically export all SDI runs to logsout-formatted .mat files

% Get all logged run IDs from SDI
runIDs = Simulink.sdi.getAllRunIDs;

for i = 1:length(runIDs)
    runObj = Simulink.sdi.getRun(runIDs(i));

    % Export to Dataset (same format as out.logsout)
    logsout = Simulink.sdi.exportRun(runObj.ID, 'Dataset');

    % Save to disk
    fname = sprintf('logsout_run_%03d.mat', i);
    save(fname, 'logsout');

    fprintf("Exported run %d to %s\n", i, fname);
end
