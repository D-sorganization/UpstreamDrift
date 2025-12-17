%Numerically Compute ZTCF and DELTA Linear Work


% ZTCF Script:
% Work is the dot product of force and velocity integrated over time.

% Create Scalar for Sample Time (Currently every 0.0001 seconds)
S=0.0001;

% Find Height
H=height(ZTCF);

for i=1:H

% Forces (Generate 1x3 array with force at time i)
FTemp=ZTCF{i,["TotalHandForceGlobal"]};
LHFTemp=ZTCF{i,["LWonClubFGlobal"]};
RHFTemp=ZTCF{i,["RWonClubFGlobal"]};
LEFTemp=ZTCF{i,["LArmonLForearmFGlobal"]};
REFTemp=ZTCF{i,["RArmonRForearmFGlobal"]};
LSFTemp=ZTCF{i,["LSonLArmFGlobal"]};
RSFTemp=ZTCF{i,["RSonRArmFGlobal"]};

F(i,1:3)=FTemp;
LHF(i,1:3)=LHFTemp;
RHF(i,1:3)=RHFTemp;
LEF(i,1:3)=LEFTemp;
REF(i,1:3)=REFTemp;
LSF(i,1:3)=LSFTemp;
RSF(i,1:3)=RSFTemp;

% Velocities
VTemp=ZTCF{i,["MidHandVelocity"]};
LHVTemp=ZTCF{i,["LeftHandVelocity"]};
RHVTemp=ZTCF{i,["RightHandVelocity"]};
LEVTemp=ZTCF{i,["LEvGlobal"]};
REVTemp=ZTCF{i,["REvGlobal"]};
LSVTemp=ZTCF{i,["LSvGlobal"]};
RSVTemp=ZTCF{i,["RSvGlobal"]};

%Dot Products
PTemp=dot(FTemp,VTemp);
LHPTemp=dot(LHFTemp,LHVTemp);
RHPTemp=dot(RHFTemp,RHVTemp);
LEPTemp=dot(LEFTemp,LEVTemp);
REPTemp=dot(REFTemp,REVTemp);
LSPTemp=dot(LSFTemp,LSVTemp);
RSPTemp=dot(RSFTemp,RSVTemp);

P(i,1)=PTemp;
LHP(i,1)=LHPTemp;
RHP(i,1)=RHPTemp;
LEP(i,1)=LEPTemp;
REP(i,1)=REPTemp;
LSP(i,1)=LSPTemp;
RSP(i,1)=RSPTemp;

end

% Cleanup
clear FTemp;
clear LHFTemp;
clear RHFTemp;
clear LEFTemp;
clear REFTemp;
clear LSFTemp;
clear RSFTemp;
clear VTemp;
clear LHVTemp;
clear RHVTemp;
clear LEVTemp;
clear REVTemp;
clear LSVTemp;
clear RSVTemp;
clear i;
clear H;
clear PTemp;
clear LHPTemp;
clear RHPTemp;
clear LEPTemp;
clear REPTemp;
clear LSPTemp;
clear RSPTemp;

% Work and Impulse Calculation
LinearWorkNumerical=S*cumtrapz(P);
LinearImpulseNumerical=S*cumtrapz(F);

LHLinearWorkNumerical=S*cumtrapz(LHP);
LHLinearImpulseNumerical=S*cumtrapz(LHF);

RHLinearWorkNumerical=S*cumtrapz(RHP);
RHLinearImpulseNumerical=S*cumtrapz(RHF);

LELinearWorkNumerical=S*cumtrapz(LEP);
LELinearImpulseNumerical=S*cumtrapz(LEF);

RELinearWorkNumerical=S*cumtrapz(REP);
RELinearImpulseNumerical=S*cumtrapz(REF);

LSLinearWorkNumerical=S*cumtrapz(LSP);
LSLinearImpulseNumerical=S*cumtrapz(LSF);

RSLinearWorkNumerical=S*cumtrapz(RSP);
RSLinearImpulseNumerical=S*cumtrapz(RSF);

% Write the values to the table
ZTCF.("LinearWorkonClub")=LinearWorkNumerical;
ZTCF.("LeftHandLinearWorkonClub")=LHLinearWorkNumerical;
ZTCF.("RightHandLinearWorkonClub")=RHLinearWorkNumerical;
ZTCF.("LELinearWorkonForearm")=LELinearWorkNumerical;
ZTCF.("RELinearWorkonForearm")=RELinearWorkNumerical;
ZTCF.("LSLinearWorkonArm")=LSLinearWorkNumerical;
ZTCF.("RSLinearWorkonArm")=RSLinearWorkNumerical;

ZTCF.("LinearImpulseonClub")=LinearImpulseNumerical;
ZTCF.("LHLinearImpulseonClub")=LHLinearImpulseNumerical;
ZTCF.("RHLinearImpulseonClub")=RHLinearImpulseNumerical;
ZTCF.("LELinearImpulseonForearm")=LELinearImpulseNumerical;
ZTCF.("RELinearImpulseonForearm")=RELinearImpulseNumerical;
ZTCF.("LSLinearImpulseonArm")=LSLinearImpulseNumerical;
ZTCF.("RSLinearImpulseonArm")=RSLinearImpulseNumerical;

% Cleanup
clear LHLinearWorkNumerical;
clear LHLinearImpulseNumerical;
clear RHLinearWorkNumerical;
clear RHLinearImpulseNumerical;
clear LinearWorkNumerical;
clear LinearImpulseNumerical;
clear LELinearWorkNumerical;
clear LELinearImpulseNumerical;
clear RELinearWorkNumerical;
clear RELinearImpulseNumerical;
clear LSLinearWorkNumerical;
clear LSLinearImpulseNumerical;
clear RSLinearWorkNumerical;
clear RSLinearImpulseNumerical;
clear F;
clear LHF;
clear RHF;
clear LEF;
clear REF;
clear LSF;
clear RSF;

clear P;
clear LHP;
clear RHP;
clear LEP;
clear REP;
clear LSP;
clear RSP;

clear S;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DELTA Script:
% Work is the dot product of force and velocity integrated over time.

% Create Scalar for Sample Time (Currently every 0.0001 seconds)
S=0.0001;

% Find Height
H=height(DELTA);

for i=1:H

% Forces (Generate 1x3 array with force at time i)
FTemp=DELTA{i,["TotalHandForceGlobal"]};
LHFTemp=DELTA{i,["LWonClubFGlobal"]};
RHFTemp=DELTA{i,["RWonClubFGlobal"]};
LEFTemp=DELTA{i,["LArmonLForearmFGlobal"]};
REFTemp=DELTA{i,["RArmonRForearmFGlobal"]};
LSFTemp=DELTA{i,["LSonLArmFGlobal"]};
RSFTemp=DELTA{i,["RSonRArmFGlobal"]};

F(i,1:3)=FTemp;
LHF(i,1:3)=LHFTemp;
RHF(i,1:3)=RHFTemp;
LEF(i,1:3)=LEFTemp;
REF(i,1:3)=REFTemp;
LSF(i,1:3)=LSFTemp;
RSF(i,1:3)=RSFTemp;

% Velocities
VTemp=ZTCF{i,["MidHandVelocity"]};
LHVTemp=ZTCF{i,["LeftHandVelocity"]};
RHVTemp=ZTCF{i,["RightHandVelocity"]};
LEVTemp=ZTCF{i,["LEvGlobal"]};
REVTemp=ZTCF{i,["REvGlobal"]};
LSVTemp=ZTCF{i,["LSvGlobal"]};
RSVTemp=ZTCF{i,["RSvGlobal"]};

%Dot Products
PTemp=dot(FTemp,VTemp);
LHPTemp=dot(LHFTemp,LHVTemp);
RHPTemp=dot(RHFTemp,RHVTemp);
LEPTemp=dot(LEFTemp,LEVTemp);
REPTemp=dot(REFTemp,REVTemp);
LSPTemp=dot(LSFTemp,LSVTemp);
RSPTemp=dot(RSFTemp,RSVTemp);

P(i,1)=PTemp;
LHP(i,1)=LHPTemp;
RHP(i,1)=RHPTemp;
LEP(i,1)=LEPTemp;
REP(i,1)=REPTemp;
LSP(i,1)=LSPTemp;
RSP(i,1)=RSPTemp;

end

% Cleanup
clear FTemp;
clear LHFTemp;
clear RHFTemp;
clear LEFTemp;
clear REFTemp;
clear LSFTemp;
clear RSFTemp;
clear VTemp;
clear LHVTemp;
clear RHVTemp;
clear LEVTemp;
clear REVTemp;
clear LSVTemp;
clear RSVTemp;
clear i;
clear H;
clear PTemp;
clear LHPTemp;
clear RHPTemp;
clear LEPTemp;
clear REPTemp;
clear LSPTemp;
clear RSPTemp;

% Work and Impulse Calculation
LinearWorkNumerical=S*cumtrapz(P);
LinearImpulseNumerical=S*cumtrapz(F);

LHLinearWorkNumerical=S*cumtrapz(LHP);
LHLinearImpulseNumerical=S*cumtrapz(LHF);

RHLinearWorkNumerical=S*cumtrapz(RHP);
RHLinearImpulseNumerical=S*cumtrapz(RHF);

LELinearWorkNumerical=S*cumtrapz(LEP);
LELinearImpulseNumerical=S*cumtrapz(LEF);

RELinearWorkNumerical=S*cumtrapz(REP);
RELinearImpulseNumerical=S*cumtrapz(REF);

LSLinearWorkNumerical=S*cumtrapz(LSP);
LSLinearImpulseNumerical=S*cumtrapz(LSF);

RSLinearWorkNumerical=S*cumtrapz(RSP);
RSLinearImpulseNumerical=S*cumtrapz(RSF);

% Write the values to the table
DELTA.("LinearWorkonClub")=LinearWorkNumerical;
DELTA.("LeftHandLinearWorkonClub")=LHLinearWorkNumerical;
DELTA.("RightHandLinearWorkonClub")=RHLinearWorkNumerical;
DELTA.("LELinearWorkonForearm")=LELinearWorkNumerical;
DELTA.("RELinearWorkonForearm")=RELinearWorkNumerical;
DELTA.("LSLinearWorkonArm")=LSLinearWorkNumerical;
DELTA.("RSLinearWorkonArm")=RSLinearWorkNumerical;

DELTA.("LinearImpulseonClub")=LinearImpulseNumerical;
DELTA.("LHLinearImpulseonClub")=LHLinearImpulseNumerical;
DELTA.("RHLinearImpulseonClub")=RHLinearImpulseNumerical;
DELTA.("LELinearImpulseonForearm")=LELinearImpulseNumerical;
DELTA.("RELinearImpulseonForearm")=RELinearImpulseNumerical;
DELTA.("LSLinearImpulseonArm")=LSLinearImpulseNumerical;
DELTA.("RSLinearImpulseonArm")=RSLinearImpulseNumerical;

% Cleanup
clear LHLinearWorkNumerical;
clear LHLinearImpulseNumerical;
clear RHLinearWorkNumerical;
clear RHLinearImpulseNumerical;
clear LinearWorkNumerical;
clear LinearImpulseNumerical;
clear LELinearWorkNumerical;
clear LELinearImpulseNumerical;
clear RELinearWorkNumerical;
clear RELinearImpulseNumerical;
clear LSLinearWorkNumerical;
clear LSLinearImpulseNumerical;
clear RSLinearWorkNumerical;
clear RSLinearImpulseNumerical;
clear F;
clear LHF;
clear RHF;
clear LEF;
clear REF;
clear LSF;
clear RSF;

clear P;
clear LHP;
clear RHP;
clear LEP;
clear REP;
clear LSP;
clear RSP;

clear S;
