%Numerically Compute ZTCF and DELTA Linear Work



%ZTCF Script:
%Work is the dot product of force and velocity integrated over time.
F=
V=
Fx=
Fy=
Fz=
Vx=
Vy=
Vz=
P=dot(F,V);
Work=S*cumtrapz


FHP=ZTCF{:,["ForceAlongHandPath"]};
dFHP=1/S*gradient(FHP);
Prod=FHP.*dFHP;

LHFHP=ZTCF{:,["LeftHandForceAlongHandPath"]};
dLHFHP=1/S*gradient(LHFHP);
LHProd=LHFHP.*dLHFHP;

RHFHP=ZTCF{:,["RightHandForceAlongHandPath"]};
dRHFHP=1/S*gradient(RHFHP);
RHProd=RHFHP.*dRHFHP;

LinearWorkNumerical=S*cumtrapz(Prod);
LHLinearWorkNumerical=S*cumtrapz(LHProd);
RHLinearWorkNumerical=S*cumtrapz(RHProd);

%Base Check
BASEFHP=BASE{:,["ForceAlongHandPath"]};
dBASEFHP=1/S*gradient(BASEFHP);
BASEProd=BASEFHP.*dBASEFHP;
BASELinearWork=S*cumtrapz(BASEProd);
BASELWFROMTABLE=BASE{:,["LinearWorkonClub"]};

ZTCF.("LinearWorkonClub")=LinearWorkNumerical;
ZTCF.("LeftHandLinearWorkonClub")=LHLinearWorkNumerical;
ZTCF.("RightHandLinearWorkonClub")=RHLinearWorkNumerical;

clear FHP;
clear dFHP;
clear Prod;
clear LHFHP;
clear dLHFHP;
clear LHProd;
clear RHFHP;
clear dRHFHP;
clear RHProd;
clear BASEFHP;
clear dBASEFHP;
clear BASEProd;
clear LHLinearWorkNumerical;
clear RHLinearWorkNumerical;
clear LinearWorkNumerical;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FHP=DELTA{:,["ForceAlongHandPath"]};
dFHP=1/S*gradient(FHP);
Prod=FHP.*dFHP;

LHFHP=DELTA{:,["LeftHandForceAlongHandPath"]};
dLHFHP=1/S*gradient(LHFHP);
LHProd=LHFHP.*dLHFHP;

RHFHP=DELTA{:,["RightHandForceAlongHandPath"]};
dRHFHP=1/S*gradient(RHFHP);
RHProd=RHFHP.*dRHFHP;

LinearWorkNumerical=S*cumtrapz(Prod);
LHLinearWorkNumerical=S*cumtrapz(LHProd);
RHLinearWorkNumerical=S*cumtrapz(RHProd);

DELTA.("LinearWorkonClub")=LinearWorkNumerical;
DELTA.("LeftHandLinearWorkonClub")=LHLinearWorkNumerical;
DELTA.("RightHandLinearWorkonClub")=RHLinearWorkNumerical;

clear S;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate the Midpoint Linear Work with Trapezoidal Integration at Each Time:
% for i=2:H
%     x=DELTA{1:i,["Time"]};
%     y=Prod(1:i,1);
%     LW=trapz(x,y);
%     LinearWorkNumerical(i,1)=LW;
% end
% 
% %Write the values to the DELTA table to replace:
% DELTA.("LinearWorkonClub")=LinearWorkNumerical;
% 
% clear x;
% clear y;
% clear LW;
% clear i;
% clear FHP;
% clear dFHP;
% clear Prod;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Calculate the Left Hand Linear Work with Trapezoidal Integration at Each Time:
% for i=2:H
%     x=DELTA{1:i,["Time"]};
%     y=LHProd(1:i,1);
%     LHLW=trapz(x,y);
%     LHLinearWorkNumerical(i,1)=LHLW;
% end
% 
% %Write the values to the DELTA table to replace:
% DELTA.("LeftHandLinearWorkonClub")=LHLinearWorkNumerical;
% 
% clear x;
% clear y;
% clear LHLW;
% clear i;
% clear LHFHP;
% clear dLHFHP;
% clear LHProd;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Calculate the Right Hand Linear Work with Trapezoidal Integration at Each Time:
% for i=2:H
%     x=DELTA{1:i,["Time"]};
%     y=RHProd(1:i,1);
%     RHLW=trapz(x,y);
%     RHLinearWorkNumerical(i,1)=RHLW;
% end
% 
% %Write the values to the DELTA table to replace:
% DELTA.("RightHandLinearWorkonClub")=RHLinearWorkNumerical;
% 
% clear x;
% clear y;
% clear H;
% clear RHLW;
% clear i;
% clear RHFHP;
% clear dRHFHP;
% clear RHProd;
% clear H;
% 
% clear LHLinearWorkNumerical;
% clear RHLinearWorkNumerical;
% clear LinearWorkNumerical;