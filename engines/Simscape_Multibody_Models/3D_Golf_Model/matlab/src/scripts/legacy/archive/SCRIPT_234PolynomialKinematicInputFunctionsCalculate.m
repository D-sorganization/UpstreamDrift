% Calculate the position, velocity, and acceleration functions for 2,3,4
% power polynominal representation.

syms Acceleration Velocity Position t A B Ao Af Vo Vf Po Pf Tt AJ AA AV

% Tt = TargetImpactTime
EquationAvgJerk=AJ==(Af-Ao)/Tt;
EquationAvgAccel=AA==(Vf-Vo)/Tt;
EquationAvgVel=AV==(Pf-Po)/Tt;

% Solve the average jerk equation for Af
finalaccel=solve([EquationAvgJerk],[Af]);
Af=finalaccel;

% Define the first two constraint equations for the initial and final positions and
% express them in terms of the AvgJerk,AvgAccel,AvgVel
ConstraintEquation1=(Af - Ao)/Tt == B + A*Tt;
ConstraintEquation2=(Vf - Vo)/Tt == (A*Tt^2)/3 + (B*Tt)/2 + Ao;
ConstraintEquation3=(Pf - Po)/Tt == (A*Tt^3)/12 + (B*Tt^2)/6 + (Ao*Tt)/2 + Vo;

% Solve Equations to determine coefficients A and B:
constants=solve([ConstraintEquation2,ConstraintEquation3],[A,B])

% Define Equations for A and B using the results of the solution.
A=constants.A
B=constants.B

% Evaluate the position, velocity, and acceleration equations with the
% solutions for Af,A, and B included. The independent variables that can be
% manipulated are Ao and AJ.
Acceleration=A*t^2+B*t+Ao
Velocity=A*t^3/3+B*t^2/2+Ao*t+Vo
Position=A*t^4/12+B*t^3/6+Ao*t^2/2+Vo*t+Po

% Calculate Jerk and Snap - Higher order derivatives that can be used to
% make a signal coming out of the function block that can be minimized in
% the trajectory optimization.
Jerk=diff(Acceleration,t);
Snap=diff(Jerk,t);
Crackle=diff(Snap,t);
Pop=diff(Crackle,t);

% expand(Acceleration);
% expand(Velocity);
% expand(Position);
