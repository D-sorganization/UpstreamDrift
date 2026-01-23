classdef data_manager < handle
    % DATA_MANAGER - Manages data sharing between tabs in the Golf Analysis App
    %
    % This class handles in-memory data passing between different tabs
    % using application data storage.
    %
    % Usage:
    %   dm = data_manager(main_fig);
    %   dm.set_simulation_data(sim_data);
    %   sim_data = dm.get_simulation_data();

    properties (Access = private)
        main_figure  % Handle to main application figure
    end

    methods
        function obj = data_manager(main_fig)
            % Constructor - initialize with main figure handle
            %
            % Inputs:
            %   main_fig - Handle to the main application figure
            %
            % Returns:
            %   obj - data_manager instance
            if nargin < 1 || ~ishandle(main_fig)
                error('data_manager:InvalidInput', 'Valid figure handle required');
            end
            obj.main_figure = main_fig;
        end

        %% Tab 1 Data (Model Setup & Simulation)
        function set_simulation_data(obj, sim_data)
            % Store simulation results from Tab 1
            setappdata(obj.main_figure, 'simulation_data', sim_data);
        end

        function sim_data = get_simulation_data(obj)
            % Retrieve simulation results for other tabs
            if isappdata(obj.main_figure, 'simulation_data')
                sim_data = getappdata(obj.main_figure, 'simulation_data');
            else
                sim_data = [];
            end
        end

        function has_data = has_simulation_data(obj)
            % Check if simulation data exists
            has_data = isappdata(obj.main_figure, 'simulation_data');
        end

        %% Tab 2 Data (ZTCF Calculation)
        function set_ztcf_data(obj, ztcf_data)
            % Store ZTCF calculation results from Tab 2
            % Expected fields: BASEQ, ZTCFQ, DELTAQ (tables)
            setappdata(obj.main_figure, 'ztcf_data', ztcf_data);
        end

        function ztcf_data = get_ztcf_data(obj)
            % Retrieve ZTCF data for visualization
            if isappdata(obj.main_figure, 'ztcf_data')
                ztcf_data = getappdata(obj.main_figure, 'ztcf_data');
            else
                ztcf_data = [];
            end
        end

        function has_data = has_ztcf_data(obj)
            % Check if ZTCF data exists
            has_data = isappdata(obj.main_figure, 'ztcf_data');
        end

        %% Tab 3 Data (Analysis & Visualization)
        function set_analysis_state(obj, state)
            % Store current analysis state from Tab 3
            setappdata(obj.main_figure, 'analysis_state', state);
        end

        function state = get_analysis_state(obj)
            % Retrieve analysis state
            if isappdata(obj.main_figure, 'analysis_state')
                state = getappdata(obj.main_figure, 'analysis_state');
            else
                state = [];
            end
        end

        %% General Data Management
        function clear_all_data(obj)
            % Clear all application data
            if ishandle(obj.main_figure)
                props = getappdata(obj.main_figure);
                fields = fieldnames(props);
                for i = 1:length(fields)
                    rmappdata(obj.main_figure, fields{i});
                end
            end
        end

        function clear_data(obj, data_name)
            % Clear specific data item
            if isappdata(obj.main_figure, data_name)
                rmappdata(obj.main_figure, data_name);
            end
        end

        function save_session(obj, filename)
            % Save all data to file for session recovery
            session_data = struct();

            if obj.has_simulation_data()
                session_data.simulation_data = obj.get_simulation_data();
            end

            if obj.has_ztcf_data()
                session_data.ztcf_data = obj.get_ztcf_data();
            end

            if ~isempty(obj.get_analysis_state())
                session_data.analysis_state = obj.get_analysis_state();
            end

            save(filename, 'session_data');
            fprintf('Session saved to: %s\n', filename);
        end

        function load_session(obj, filename)
            % Load session data from file
            if ~exist(filename, 'file')
                error('data_manager:FileNotFound', 'Session file not found: %s', filename);
            end

            loaded = load(filename);
            session_data = loaded.session_data;

            if isfield(session_data, 'simulation_data')
                obj.set_simulation_data(session_data.simulation_data);
            end

            if isfield(session_data, 'ztcf_data')
                obj.set_ztcf_data(session_data.ztcf_data);
            end

            if isfield(session_data, 'analysis_state')
                obj.set_analysis_state(session_data.analysis_state);
            end

            fprintf('Session loaded from: %s\n', filename);
        end

        function info = get_data_info(obj)
            % Get summary of available data
            info = struct();
            info.has_simulation = obj.has_simulation_data();
            info.has_ztcf = obj.has_ztcf_data();
            info.has_analysis_state = ~isempty(obj.get_analysis_state());

            % Get size information if available
            if info.has_ztcf
                ztcf_data = obj.get_ztcf_data();
                if isstruct(ztcf_data)
                    if isfield(ztcf_data, 'BASEQ')
                        info.num_frames = height(ztcf_data.BASEQ);
                    end
                end
            end
        end
    end
end
