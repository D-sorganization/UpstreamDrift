% File: matlab/run_all.m
function run_all()
    % RUN_ALL Orchestrates the full Golf Model simulation pipeline.
    %
    % This script:
    % 1. Configures the environment and reproducibility.
    % 2. Validates dependencies (PhysicsConstants).
    % 3. Runs a baseline ballistic simulation using canonical constants.
    % 4. Generates artifact plots and metadata.
    %
    % Output:
    %   All artifacts are saved to output/<date>/baseline/

    % 1) Configure Environment and Reproducibility
    rng(42);
    cleanup = onCleanup(@() cleanupPath());
    
    % Add source paths
    baseDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(baseDir, 'src', 'classes'));
    
    % Prepare output directory
    timestamp = datestr(datetime('now'), 'yyyy-mm-dd_HHMMSS');
    outdir = fullfile(baseDir, '..', 'output', timestamp, 'baseline');
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end
    
    fprintf('Starting Golf Model Pipeline...\n');
    fprintf('Output Directory: %s\n', outdir);

    % 2) Validate Dependencies
    try
        phys = PhysicsConstants.getPhysicsConstants();
        fprintf('PhysicsConstants loaded successfully.\n');
    catch ME
        error('DependencyError:FailedToLoadConstants', ...
              'Could not load PhysicsConstants. Ensure src/classes is on path.\n%s', ME.message);
    end

    % 3) Run Baseline Simulation (Ballistic Trajectory)
    % Using simple projectile motion with drag for demonstration/baseline
    fprintf('Running baseline simulation...\n');
    
    v0 = phys.TYPICAL_BALL_SPEED_MS; % m/s
    launchAngleDeg = 12.0; % typical launch
    theta = launchAngleDeg * (pi/180);
    
    g = phys.GRAVITY_EARTH;
    rho = phys.AIR_DENSITY_SEA_LEVEL;
    area = pi * (phys.GOLF_BALL_RADIUS_M)^2;
    Cd = phys.GOLF_BALL_DRAG_COEFFICIENT;
    m = phys.GOLF_BALL_MASS_KG;
    
    % Initial state
    vx = v0 * cos(theta);
    vy = v0 * sin(theta);
    x = 0; y = 0;
    
    dt = phys.DEFAULT_TIMESTEP_S;
    t = 0;
    
    history.time = [];
    history.x = [];
    history.y = [];
    history.vx = [];
    history.vy = [];
    
    while y >= 0 && t < phys.MAX_SIMULATION_TIME_S
        % Store state
        history.time(end+1) = t;
        history.x(end+1) = x;
        history.y(end+1) = y;
        history.vx(end+1) = vx;
        history.vy(end+1) = vy;
        
        % Drag force
        v = sqrt(vx^2 + vy^2);
        Fd = 0.5 * rho * v^2 * area * Cd;
        if v > 0
            Fdx = -Fd * (vx/v);
            Fdy = -Fd * (vy/v);
        else
            Fdx = 0; Fdy = 0;
        end
        
        % Update state (Euler integration for simplicity in baseline)
        ax = Fdx / m;
        ay = (Fdy / m) - g;
        
        vx = vx + ax * dt;
        vy = vy + ay * dt;
        
        x = x + vx * dt;
        y = y + vy * dt;
        
        t = t + dt;
    end
    
    fprintf('Simulation complete. Range: %.2f m\n', x);

    % 4) Generate Artifacts
    % Plot Trajectory
    fig = figure('Visible', 'off');
    plot(history.x, history.y, 'LineWidth', 2);
    xlabel('Distance (m)');
    ylabel('Height (m)');
    title(sprintf('Baseline Trajectory (Range: %.1f m)', x));
    grid on;
    
    plotFile = fullfile(outdir, 'trajectory_baseline.png');
    saveas(fig, plotFile);
    close(fig);
    fprintf('Saved plot to %s\n', plotFile);
    
    % Save Metadata
    meta.date = datestr(datetime('now'));
    meta.matlab_version = version;
    meta.commit_sha = getenv('CI_COMMIT_SHA');
    if isempty(meta.commit_sha)
        meta.commit_sha = 'local-dev';
    end
    meta.physics_source = 'PhysicsConstants.m';
    meta.simulation_type = '2D Ballistic with Drag';
    meta.results.range_m = x;
    meta.results.max_height_m = max(history.y);
    meta.results.flight_time_s = t;
    
    fid = fopen(fullfile(outdir, 'metadata.json'), 'w');
    fprintf(fid, '%s', jsonencode(meta, 'PrettyPrint', true));
    fclose(fid);
    
    fprintf('Pipeline finished successfully.\n');
end

function cleanupPath()
    % Optional cleanup if needed (path usually persists in session, 
    % but good hygiene to mention)
end
