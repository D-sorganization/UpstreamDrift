classdef PhysicsConstants
    % PHYSICSCONSTANTS Standard physical constants for golf swing analysis
    %
    % This class defines physical constants used in biomechanical modeling
    % and golf swing simulations. All values include units, sources, and
    % uncertainty estimates where applicable.
    %
    % Usage:
    %   g = PhysicsConstants.GRAVITY_EARTH;
    %   clubMass = PhysicsConstants.DRIVER_MASS_KG;
    %
    % Unit System:
    %   - SI units preferred (kg, m, s, N, J, W)
    %   - Imperial units provided where industry standard
    %   - Conversion factors included for convenience
    %
    % References:
    %   - USGA Rules of Golf (equipment specifications)
    %   - NIST CODATA (fundamental constants)
    %   - Biomechanics literature (human performance)
    %
    % See also: UICOLORS, GUILAYOUTCONSTANTS

    properties (Constant)
        %% Universal Physical Constants

        % GRAVITY_EARTH - Standard gravity at sea level [m/s²]
        % Source: NIST CODATA 2018
        % Uncertainty: ±0.00001 m/s²
        GRAVITY_EARTH = 9.80665

        % AIR_DENSITY_SEA_LEVEL - Air density at sea level, 20°C [kg/m³]
        % Source: ISO 2533:1975 Standard Atmosphere
        % Conditions: T=293.15K (20°C), P=101325 Pa, 50% humidity
        AIR_DENSITY_SEA_LEVEL = 1.204

        %% Golf Ball Properties

        % GOLF_BALL_MASS_KG - Maximum golf ball mass [kg]
        % Source: USGA Rule 5-1 (R&A Equipment Rules)
        % Imperial: 1.620 oz maximum
        % Typical tour ball: 45.6-45.93g (within regulation)
        GOLF_BALL_MASS_KG = 0.04593

        % GOLF_BALL_DIAMETER_M - Minimum golf ball diameter [m]
        % Source: USGA Rule 5-1
        % Imperial: 1.680 inches minimum
        % Typical: 42.67 mm (1.680")
        GOLF_BALL_DIAMETER_M = 0.04267

        % GOLF_BALL_RADIUS_M - Golf ball radius [m]
        % Calculated from diameter
        GOLF_BALL_RADIUS_M = 0.04267 / 2

        % GOLF_BALL_DRAG_COEFFICIENT - Typical drag coefficient (dimensionless)
        % Source: Bearman & Harvey (1976), Erlichson (1983)
        % Range: 0.22-0.28 depending on speed and spin
        % Value: 0.25 (representative average for moderate speeds)
        GOLF_BALL_DRAG_COEFFICIENT = 0.25

        % GOLF_BALL_LIFT_COEFFICIENT - Typical lift coefficient (dimensionless)
        % Source: Bearman & Harvey (1976)
        % Depends on spin rate and speed
        % Value: 0.15 (typical for backspin)
        GOLF_BALL_LIFT_COEFFICIENT = 0.15

        %% Golf Club Properties (Driver)

        % DRIVER_MASS_KG - Typical driver total mass [kg]
        % Source: Industry measurements, USGA data
        % Range: 280-320g for modern drivers
        % Value: 0.310 kg (310g - middle of typical range)
        DRIVER_MASS_KG = 0.310

        % DRIVER_LENGTH_M - Typical driver length [m]
        % Source: USGA Rule 4-1d (maximum 48 inches)
        % Imperial: 45-46 inches typical
        % Value: 1.1684 m (46 inches)
        DRIVER_LENGTH_M = 1.1684

        % DRIVER_SHAFT_MASS_KG - Typical driver shaft mass [kg]
        % Source: Manufacturer specifications
        % Range: 50-80g for graphite shafts
        % Value: 0.065 kg (65g - typical stiff flex graphite)
        DRIVER_SHAFT_MASS_KG = 0.065

        % DRIVER_HEAD_MASS_KG - Typical driver head mass [kg]
        % Source: USGA rules, manufacturer data
        % Range: 190-210g (modern 460cc drivers)
        % Value: 0.200 kg (200g)
        DRIVER_HEAD_MASS_KG = 0.200

        % DRIVER_LOFT_DEG - Typical driver loft [degrees]
        % Source: Tour averages
        % Range: 8-12° (tour average ~10.5°)
        % Value: 10.5°
        DRIVER_LOFT_DEG = 10.5

        %% Golf Club Properties (Iron)

        % IRON_7_MASS_KG - Typical 7-iron total mass [kg]
        % Source: Industry measurements
        % Value: 0.420 kg (420g typical)
        IRON_7_MASS_KG = 0.420

        % IRON_7_LENGTH_M - Typical 7-iron length [m]
        % Imperial: 37 inches typical
        % Value: 0.9398 m (37 inches)
        IRON_7_LENGTH_M = 0.9398

        % IRON_7_LOFT_DEG - Typical 7-iron loft [degrees]
        % Source: Modern game improvement irons
        % Range: 28-34° (average ~31°)
        % Value: 31°
        IRON_7_LOFT_DEG = 31

        %% Human Biomechanics

        % TYPICAL_MALE_MASS_KG - Reference male golfer mass [kg]
        % Source: Anthropometric data, tour averages
        % Value: 80 kg (176 lbs)
        TYPICAL_MALE_MASS_KG = 80

        % TYPICAL_FEMALE_MASS_KG - Reference female golfer mass [kg]
        % Source: Anthropometric data, tour averages
        % Value: 65 kg (143 lbs)
        TYPICAL_FEMALE_MASS_KG = 65

        % MAX_HUMAN_POWER_OUTPUT_W - Maximum sustainable power output [W]
        % Source: Biomechanics literature (trained athletes)
        % Duration: ~3 seconds (golf swing duration)
        % Value: 1500 W (elite male athlete, short burst)
        MAX_HUMAN_POWER_OUTPUT_W = 1500

        %% Typical Golf Swing Parameters

        % TYPICAL_CHS_MS - Typical clubhead speed (male amateur) [m/s]
        % Source: Trackman database
        % Imperial: ~90 mph
        % Value: 40.2 m/s (90 mph)
        TYPICAL_CHS_MS = 40.2

        % TYPICAL_CHS_TOUR_MS - Typical tour pro clubhead speed [m/s]
        % Source: PGA Tour averages
        % Imperial: ~113 mph
        % Value: 50.5 m/s (113 mph)
        TYPICAL_CHS_TOUR_MS = 50.5

        % TYPICAL_BALL_SPEED_MS - Typical ball speed (male amateur) [m/s]
        % Source: Trackman data
        % Imperial: ~135 mph
        % Value: 60.4 m/s (135 mph)
        TYPICAL_BALL_SPEED_MS = 60.4

        % TYPICAL_SWING_DURATION_S - Typical downswing duration [s]
        % Source: 3D motion capture studies
        % Range: 0.2-0.3s
        % Value: 0.25s (250 milliseconds)
        TYPICAL_SWING_DURATION_S = 0.25

        %% Simulation Parameters

        % DEFAULT_TIMESTEP_S - Default simulation timestep [s]
        % Value: 0.001s (1 millisecond)
        % Rationale: Captures swing dynamics at 1000 Hz
        DEFAULT_TIMESTEP_S = 0.001

        % MAX_SIMULATION_TIME_S - Maximum simulation time [s]
        % Value: 5.0s (covers full swing + follow-through)
        MAX_SIMULATION_TIME_S = 5.0

        %% Unit Conversion Factors

        % MPH_TO_MS - Miles per hour to meters per second
        % Formula: mph * 0.44704 = m/s
        MPH_TO_MS = 0.44704

        % MS_TO_MPH - Meters per second to miles per hour
        % Formula: m/s * 2.23694 = mph
        MS_TO_MPH = 2.23694

        % INCHES_TO_M - Inches to meters
        % Formula: inches * 0.0254 = m
        INCHES_TO_M = 0.0254

        % M_TO_INCHES - Meters to inches
        % Formula: m * 39.3701 = inches
        M_TO_INCHES = 39.3701

        % LBS_TO_KG - Pounds to kilograms
        % Formula: lbs * 0.453592 = kg
        LBS_TO_KG = 0.453592

        % KG_TO_LBS - Kilograms to pounds
        % Formula: kg * 2.20462 = lbs
        KG_TO_LBS = 2.20462

        % DEG_TO_RAD - Degrees to radians
        % Formula: deg * (π/180)
        DEG_TO_RAD = pi / 180

        % RAD_TO_DEG - Radians to degrees
        % Formula: rad * (180/π)
        RAD_TO_DEG = 180 / pi
    end

    methods (Static)
        function constants = getPhysicsConstants()
            % GETPHYSICSCONSTANTS Returns complete physics constants as struct
            %
            % Returns:
            %   constants - Struct containing all physics constants
            %
            % Example:
            %   phys = PhysicsConstants.getPhysicsConstants();
            %   ballMass = phys.GOLF_BALL_MASS_KG;

            constants = struct();

            % Universal constants
            constants.GRAVITY_EARTH = PhysicsConstants.GRAVITY_EARTH;
            constants.AIR_DENSITY_SEA_LEVEL = PhysicsConstants.AIR_DENSITY_SEA_LEVEL;

            % Golf ball
            constants.GOLF_BALL_MASS_KG = PhysicsConstants.GOLF_BALL_MASS_KG;
            constants.GOLF_BALL_DIAMETER_M = PhysicsConstants.GOLF_BALL_DIAMETER_M;
            constants.GOLF_BALL_RADIUS_M = PhysicsConstants.GOLF_BALL_RADIUS_M;
            constants.GOLF_BALL_DRAG_COEFFICIENT = PhysicsConstants.GOLF_BALL_DRAG_COEFFICIENT;
            constants.GOLF_BALL_LIFT_COEFFICIENT = PhysicsConstants.GOLF_BALL_LIFT_COEFFICIENT;

            % Driver
            constants.DRIVER_MASS_KG = PhysicsConstants.DRIVER_MASS_KG;
            constants.DRIVER_LENGTH_M = PhysicsConstants.DRIVER_LENGTH_M;
            constants.DRIVER_SHAFT_MASS_KG = PhysicsConstants.DRIVER_SHAFT_MASS_KG;
            constants.DRIVER_HEAD_MASS_KG = PhysicsConstants.DRIVER_HEAD_MASS_KG;
            constants.DRIVER_LOFT_DEG = PhysicsConstants.DRIVER_LOFT_DEG;

            % 7-iron
            constants.IRON_7_MASS_KG = PhysicsConstants.IRON_7_MASS_KG;
            constants.IRON_7_LENGTH_M = PhysicsConstants.IRON_7_LENGTH_M;
            constants.IRON_7_LOFT_DEG = PhysicsConstants.IRON_7_LOFT_DEG;

            % Human biomechanics
            constants.TYPICAL_MALE_MASS_KG = PhysicsConstants.TYPICAL_MALE_MASS_KG;
            constants.TYPICAL_FEMALE_MASS_KG = PhysicsConstants.TYPICAL_FEMALE_MASS_KG;
            constants.MAX_HUMAN_POWER_OUTPUT_W = PhysicsConstants.MAX_HUMAN_POWER_OUTPUT_W;

            % Swing parameters
            constants.TYPICAL_CHS_MS = PhysicsConstants.TYPICAL_CHS_MS;
            constants.TYPICAL_CHS_TOUR_MS = PhysicsConstants.TYPICAL_CHS_TOUR_MS;
            constants.TYPICAL_BALL_SPEED_MS = PhysicsConstants.TYPICAL_BALL_SPEED_MS;
            constants.TYPICAL_SWING_DURATION_S = PhysicsConstants.TYPICAL_SWING_DURATION_S;

            % Simulation
            constants.DEFAULT_TIMESTEP_S = PhysicsConstants.DEFAULT_TIMESTEP_S;
            constants.MAX_SIMULATION_TIME_S = PhysicsConstants.MAX_SIMULATION_TIME_S;
        end

        function ms = mphToMs(mph)
            % MPHTOMS Convert miles per hour to meters per second
            %
            % Args:
            %   mph - Speed in miles per hour
            %
            % Returns:
            %   ms - Speed in meters per second
            %
            % Example:
            %   chsMs = PhysicsConstants.mphToMs(113);
            %   % Returns: 50.5 m/s

            arguments
                mph double {mustBeNonnegative}
            end

            ms = mph * PhysicsConstants.MPH_TO_MS;
        end

        function mph = msToMph(ms)
            % MSTOMPH Convert meters per second to miles per hour
            %
            % Args:
            %   ms - Speed in meters per second
            %
            % Returns:
            %   mph - Speed in miles per hour
            %
            % Example:
            %   chsMph = PhysicsConstants.msToMph(50.5);
            %   % Returns: 113 mph

            arguments
                ms double {mustBeNonnegative}
            end

            mph = ms * PhysicsConstants.MS_TO_MPH;
        end
    end
end
