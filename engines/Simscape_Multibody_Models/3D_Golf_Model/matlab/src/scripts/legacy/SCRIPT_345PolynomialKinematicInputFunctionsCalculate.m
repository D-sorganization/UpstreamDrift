% Calculate the position, velocity, and acceleration functions for 2,3,4
% power polynominal representation.

syms Acceleration Velocity Position t A B Ao Af Vo Vf Po Pf Tt AJ AA AV Acc Vel Pos C D E F

% Define the equations for acceleration, velocity, and position
Acc=A*t^3+B*t^2+C*t+D;
Vel=int(Acc)+E;
Pos=int(Vel)+F;

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
constants=solve([Acco Accf Velo Velf Poso Posf],[A B C D E F]);
A=constants.A;
B=constants.B;
C=constants.C;
D=constants.D;
E=constants.E;
F=constants.F;

Acceleration=A*t^3 + B*t^2 + C*t + D
Velocity=(A*t^4)/4 + (B*t^3)/3 + (C*t^2)/2 + D*t + E
Position=(A*t^5)/20 + (B*t^4)/12 + (C*t^3)/6 + (D*t^2)/2 + E*t + F
%
% Calculate Jerk and Snap - Higher order derivatives that can be used to
% make a signal coming out of the function block that can be minimized in
% the trajectory optimization.
Jerk=gradient(Acceleration,t);
Snap=gradient(Jerk,t);
Crackle=gradient(Snap,t);
Pop=gradient(Crackle,t);

EquationAvgJerk=AJ==(Af-Ao)/Tt;
EquationAvgAccel=AA==(Vf-Vo)/Tt;
EquationAvgVel=AV==(Pf-Po)/Tt;
