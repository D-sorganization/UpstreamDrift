function golf_suite_help()
%GOLF_SUITE_HELP Display help for Golf Modeling Suite
%
% This function displays available functions and usage information
% for the Golf Modeling Suite.
%
% Usage:
%   golf_suite_help()

    fprintf('\n');
    fprintf('=== Golf Modeling Suite - Available Functions ===\n');
    fprintf('\n');
    
    fprintf('Setup and Configuration:\n');
    fprintf('  setup_golf_suite()           - Initialize MATLAB environment\n');
    fprintf('  golf_suite_paths()           - Display all suite paths\n');
    fprintf('  validate_golf_environment()  - Check toolbox availability\n');
    fprintf('\n');
    
    fprintf('Data Management:\n');
    fprintf('  load_golf_data(filename)     - Load golf swing data\n');
    fprintf('  save_golf_data(data, file)   - Save golf swing data\n');
    fprintf('  standardize_data(data)       - Standardize data format\n');
    fprintf('\n');
    
    fprintf('Analysis Functions:\n');
    fprintf('  analyze_joint_angles(data)   - Joint angle analysis\n');
    fprintf('  calculate_club_speed(data)   - Club head speed calculation\n');
    fprintf('  plot_swing_trajectory(data)  - Trajectory visualization\n');
    fprintf('\n');
    
    fprintf('Engine-Specific Launchers:\n');
    fprintf('  launch_2d_model()            - Launch 2D MATLAB model\n');
    fprintf('  launch_3d_model()            - Launch 3D MATLAB model\n');
    fprintf('  launch_pendulum_model()      - Launch pendulum models\n');
    fprintf('\n');
    
    fprintf('Utilities:\n');
    fprintf('  convert_units(val, from, to) - Unit conversions\n');
    fprintf('  export_to_python(data, file) - Export data for Python engines\n');
    fprintf('  import_from_python(file)     - Import Python engine results\n');
    fprintf('\n');
    
    fprintf('For detailed help on any function, use: help function_name\n');
    fprintf('For engine-specific documentation, see:\n');
    fprintf('  engines/matlab_simscape/2d_model/docs/\n');
    fprintf('  engines/matlab_simscape/3d_biomechanical/docs/\n');
    fprintf('\n');
end