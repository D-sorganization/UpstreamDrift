classdef ParforProgressbar < handle
    % PARFORPROGRESSBAR Progress bar for parfor loops
    %
    % Simple progress bar that works with parfor loops using DataQueue
    %
    % Usage:
    %   ppm = ParforProgressbar(N, 'Title', 'Processing');
    %   parfor i = 1:N
    %       % ... do work ...
    %       ppm.increment();
    %   end
    %   delete(ppm);
    %
    % Author: Optimized Golf Swing Analysis System
    % Date: 2025

    properties (Access = private)
        Queue
        Total
        Current
        StartTime
        Title
        UpdateInterval
        LastUpdate
    end

    properties (Constant, Access = private)
        BAR_LENGTH = 40;
        SECONDS_IN_HOUR = 3600;
        SECONDS_IN_MINUTE = 60;
    end

    methods
        function obj = ParforProgressbar(total, options)
            % Constructor
            arguments
                total (1,1) double
                options.Title (1,:) char = 'Progress'
                options.UpdateInterval (1,1) double = 0.1
            end

            obj.Total = total;
            obj.Current = 0;
            obj.Title = options.Title;
            obj.UpdateInterval = options.UpdateInterval;
            obj.StartTime = tic;
            obj.LastUpdate = 0;

            % Create DataQueue for parallel communication
            obj.Queue = parallel.pool.DataQueue;
            afterEach(obj.Queue, @(~) obj.updateProgress());

            % Display initial progress
            obj.displayProgress();
        end

        function increment(obj, ~)
            % Increment progress counter
            arguments
                obj
                ~
            end
            send(obj.Queue, true);
        end

        function delete(obj)
            % Destructor - display final progress
            arguments
                obj
            end
            obj.displayProgress();
            fprintf('\n');
        end
    end

    methods (Access = private)
        function updateProgress(obj)
            % Update progress counter and display
            arguments
                obj
            end
            obj.Current = obj.Current + 1;

            % Only update display at specified intervals
            elapsed = toc(obj.StartTime);
            if elapsed - obj.LastUpdate >= obj.UpdateInterval || ...
               obj.Current == obj.Total
                obj.displayProgress();
                obj.LastUpdate = elapsed;
            end
        end

        function displayProgress(obj)
            % Display progress bar
            arguments
                obj
            end
            percent = obj.Current / obj.Total * 100;
            elapsed = toc(obj.StartTime);

            % Estimate remaining time
            if obj.Current > 0
                rate = obj.Current / elapsed;
                remaining = (obj.Total - obj.Current) / rate;
                eta_str = sprintf('ETA: %s', obj.formatTime(remaining));
            else
                eta_str = 'ETA: --:--';
            end

            % Create progress bar
            bar_length = obj.BAR_LENGTH;
            filled = round(bar_length * obj.Current / obj.Total);
            bar = ['[' repmat('=', 1, filled) repmat(' ', 1, bar_length - filled) ']'];

            % Display
            fprintf('\r   %s %s %3.0f%% (%d/%d) | %s | Elapsed: %s', ...
                obj.Title, bar, percent, obj.Current, obj.Total, ...
                eta_str, obj.formatTime(elapsed));
        end

        function str = formatTime(obj, seconds)
            % Format time duration as HH:MM:SS
            arguments
                obj
                seconds
            end
            hours = floor(seconds / obj.SECONDS_IN_HOUR);
            minutes = floor(mod(seconds, obj.SECONDS_IN_HOUR) / obj.SECONDS_IN_MINUTE);
            secs = floor(mod(seconds, obj.SECONDS_IN_MINUTE));

            if hours > 0
                str = sprintf('%02d:%02d:%02d', hours, minutes, secs);
            else
                str = sprintf('%02d:%02d', minutes, secs);
            end
        end
    end
end
