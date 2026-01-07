% CONSTANTS  Physical and mathematical constants for MATLAB code
%   This file defines standard constants used throughout the MATLAB codebase
%   to avoid magic numbers and ensure consistency.
%
%   Usage:
%       constants;  % Load constants into workspace
%       g = GRAVITY_STANDARD_M_S2;
%
%   See also: run_all, aba, rnea, crba

% Physical constants (NIST reference values)
% Note: These are constant definitions, not magic numbers
GRAVITY_STANDARD_M_S2 = 9.80665;  % Standard gravity [m/s²], NIST reference value

% Spatial algebra constants
% Note: These are constant definitions for spatial vector dimensions
SPATIAL_DIM = 6;  % Dimension of spatial vectors (3 linear + 3 angular)
SPATIAL_LIN_DIM = 3;  % Linear dimension (position/velocity)
SPATIAL_ANG_DIM = 3;  % Angular dimension (orientation/angular velocity)

% Mathematical constants
PI = pi;  % π constant [dimensionless]
PI_HALF = pi / 2;  % π/2 constant [dimensionless]
PI_QUARTER = pi / 4;  % π/4 constant [dimensionless]

% Default random seed for reproducibility
% Note: This is a constant definition for test reproducibility
DEFAULT_RNG_SEED = 42;  % Default random number generator seed (commonly used value; any fixed value ensures reproducibility)

% Common test/model parameters (with units documented)
% Link lengths [m]
LINK_LENGTH_1_M = 1.0;  % Link 1 length [m]
LINK_LENGTH_2_M = 0.8;  % Link 2 length [m]

% Masses [kg]
MASS_1_KG = 1.0;  % Mass 1 [kg]
MASS_2_KG = 0.8;  % Mass 2 [kg]

% Test configuration angles [rad]
TEST_ANGLE_1_RAD = 0.3;  % Test angle 1 [rad]
TEST_ANGLE_2_RAD = 0.2;  % Test angle 2 [rad]
TEST_ANGLE_3_RAD = 1.5;  % Test angle 3 [rad]

% Test velocities [rad/s or m/s]
TEST_VELOCITY_1 = 0.5;  % Test velocity 1 [rad/s]
TEST_VELOCITY_2 = 0.3;  % Test velocity 2 [rad/s]

% Test accelerations [rad/s² or m/s²]
TEST_ACCELERATION_1 = 0.2;  % Test acceleration 1 [rad/s²]
TEST_ACCELERATION_2 = 0.1;  % Test acceleration 2 [rad/s²]

% Test torques/forces [N⋅m or N]
TEST_TORQUE_1_NM = 1.0;  % Test torque 1 [N⋅m]
TEST_TORQUE_2_NM = 0.5;  % Test torque 2 [N⋅m]

% Test positions [m]
TEST_POSITION_1_M = 0.3;  % Test position 1 [m]
TEST_POSITION_2_M = 0.2;  % Test position 2 [m]
TEST_POSITION_3_M = 0.05;  % Test position 3 [m]

% Test angles in degrees [deg]
TEST_ANGLE_45_DEG = 45;  % Test angle 45 degrees [deg]

% Array size constants
ARRAY_SIZE_6 = 6;  % Array size 6 (spatial dimension)
ARRAY_SIZE_7 = 7;  % Array size 7 (quaternion + position)
ARRAY_SIZE_8 = 8;  % Array size 8
ARRAY_SIZE_9 = 9;  % Array size 9
ARRAY_SIZE_12 = 12;  % Array size 12

