Sna% Calculate the position, velocity, and acceleration functions for 2,3,4
% power polynominal representation.

syms Acceleration Velocity Position t A B Ao Af Vo Vf Po Pf Tt AJ AA AV Acc Vel Pos C D E F

% Define the equations for acceleration, velocity, and position
Acc=A*t^2+B*t+C;
Vel=int(Acc)+D;
Pos=int(Vel)+E;

% Substitute constraints into the equation for the initial and final
% conditions.
syms Acco Velo Poso Accf Velf Posf
Acco=Ao==subs(Acc,t,0);
Velo=Vo==subs(Vel,t,0);
Poso=Po==subs(Pos,t,0);
Accf=Af==subs(Acc,t,Tt);
Velf=Vf==subs(Vel,t,Tt);
Posf=Pf==subs(Pos,t,Tt);

% Solve for the constants
constants=solve([Acco Velo Velf Poso Posf],[A B C D E]);
A=constants.A;
B=constants.B;
C=constants.C;
D=constants.D;
E=constants.E;

Acceleration=A*t^2 + B*t + C
Velocity=(A*t^3)/3 + (B*t^2)/2 + C*t + D
Position=(A*t^4)/12 + (B*t^3)/6 + (C*t^2)/2 + D*t + E
%
% Calculate Jerk and Snap - Higher order derivatives that can be used to
% make a signal coming out of the function block that can be minimized in
% the trajectory optimization.
Jerk=gradient(Acceleration,t);
Snap=gradient(Jerk,t);
Crackle=gradient(Snap,t);
Pop=gradient(Crackle,t);
%
% Tt = TargetImpactTime
% EquationAvgJerk=AJ==(Af-Ao)/Tt;
% EquationAvgAccel=AA==(Vf-Vo)/Tt;
% EquationAvgVel=AV==(Pf-Po)/Tt;
