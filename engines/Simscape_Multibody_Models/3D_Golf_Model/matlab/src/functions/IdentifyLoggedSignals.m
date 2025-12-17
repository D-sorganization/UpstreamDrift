% Identify Logged Signals (Option to Delete)

% Load model if not already open
if ~bdIsLoaded('GolfSwing3D_KineticallyDriven')
    load_system('GolfSwing3D_KineticallyDriven');
end

% Find all lines with signal logging enabled
loggedLines = find_system('GolfSwing3D_KineticallyDriven', 'FindAll', 'on', 'Type', 'line', 'SignalLogging', 'on');

% % Disable signal logging for each
% for i = 1:length(loggedLines)
%     set(loggedLines(i), 'SignalLogging', 'off');
% end
%
% % Save the model (optional)
% save_system('GolfSwing3D_KineticallyDriven');

fprintf('Logged Signals', loggedLines);
