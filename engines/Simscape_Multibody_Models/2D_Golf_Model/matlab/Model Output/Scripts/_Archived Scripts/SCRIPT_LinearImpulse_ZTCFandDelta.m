%Numerically Compute ZTCF and DELTA Linear Impulse

%ZTCF Script:
%Generate an Array to Overwrite:
LinearImpulseNumerical=ZTCF{:,["LinearImpulseOnClub"]};
LinearImpulseNumerical(1,1)=0;

LHLinearImpulseNumerical=ZTCF{:,["LHLinearImpulseonClub"]};
LHLinearImpulseNumerical(1,1)=0;

RHLinearImpulseNumerical=ZTCF{:,["RHLinearImpulseonClub"]};
RHLinearImpulseNumerical(1,1)=0;

H=height(ZTCF);

%Generate force on hand path vectors
FHP=ZTCF{:,["ForceAlongHandPath"]};
LHFHP=ZTCF{:,["LeftHandForceAlongHandPath"]};
RHFHP=ZTCF{:,["RightHandForceAlongHandPath"]};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate the Midpoint Linear Impulse with Trapezoidal Integration at Each Time:
for i=2:H
    x=ZTCF{1:i,["Time"]};
    y=FHP(1:i,1);
    LI=trapz(x,y);
    LinearImpulseNumerical(i,1)=LI;
end

%Write the values to the ZTCF table to replace:
ZTCF.("LinearImpulseOnClub")=LinearImpulseNumerical;

clear x;
clear y;
clear LI;
clear i;
clear FHP;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate the Left Hand Linear Impulse with Trapezoidal Integration at Each Time:
for i=2:H
    x=ZTCF{1:i,["Time"]};
    y=LHFHP(1:i,1);
    LHLI=trapz(x,y);
    LHLinearImpulseNumerical(i,1)=LHLI;
end

%Write the values to the ZTCF table to replace:
ZTCF.("LeftHandLinearImpulseonClub")=LHLinearImpulseNumerical;

clear x;
clear y;
clear LHLI;
clear i;
clear LHFHP;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate the Left Hand Linear Impulse with Trapezoidal Integration at Each Time:
for i=2:H
    x=ZTCF{1:i,["Time"]};
    y=RHFHP(1:i,1);
    RHLI=trapz(x,y);
    RHLinearImpulseNumerical(i,1)=RHLI;
end

%Write the values to the ZTCF table to replace:
ZTCF.("RightHandLinearImpulseonClub")=RHLinearImpulseNumerical;

clear x;
clear y;
clear H;
clear RHLI;
clear i;
clear H;

clear LinearImpulseNumerical;
clear LHLinearImpulseNumerical;
clear RHLinearImpulseNumerical;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Delta Script:
%Generate an Array to Overwrite:
LinearImpulseNumerical=DELTA{:,["LinearImpulseOnClub"]};
LinearImpulseNumerical(1,1)=0;

LHLinearImpulseNumerical=DELTA{:,["LHLinearImpulseonClub"]};
LHLinearImpulseNumerical(1,1)=0;

RHLinearImpulseNumerical=DELTA{:,["RHLinearImpulseonClub"]};
RHLinearImpulseNumerical(1,1)=0;

H=height(DELTA);

%Generate create force on hand path vectors, derivative of force on hand
%path vector, and vector that is the product of the two. The product is
%what is integrated with respect to time in the loop that calculates linear
%Impulse.
FHP=DELTA{:,["ForceAlongHandPath"]};
LHFHP=DELTA{:,["LeftHandForceAlongHandPath"]};
RHFHP=DELTA{:,["RightHandForceAlongHandPath"]};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate the Midpoint Linear Impulse with Trapezoidal Integration at Each Time:
for i=2:H
    x=DELTA{1:i,["Time"]};
    y=FHP(1:i,1);
    LI=trapz(x,y);
    LinearImpulseNumerical(i,1)=LI;
end

%Write the values to the DELTA table to replace:
DELTA.("LinearImpulseOnClub")=LinearImpulseNumerical;

clear x;
clear y;
clear LI;
clear i;
clear FHP;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate the Left Hand Linear Impulse with Trapezoidal Integration at Each Time:
for i=2:H
    x=DELTA{1:i,["Time"]};
    y=LHFHP(1:i,1);
    LHLI=trapz(x,y);
    LHLinearImpulseNumerical(i,1)=LHLI;
end

%Write the values to the DELTA table to replace:
DELTA.("LeftHandLinearImpulseonClub")=LHLinearImpulseNumerical;

clear x;
clear y;
clear LHLI;
clear i;
clear LHFHP;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate the Left Hand Linear Impulse with Trapezoidal Integration at Each Time:
for i=2:H
    x=DELTA{1:i,["Time"]};
    y=RHFHP(1:i,1);
    RHLI=trapz(x,y);
    RHLinearImpulseNumerical(i,1)=RHLI;
end

%Write the values to the DELTA table to replace:
DELTA.("RightHandLinearImpulseonClub")=RHLinearImpulseNumerical;

clear x;
clear y;
clear H;
clear RHLI;
clear i;
clear RHFHP;
clear H;

clear LinearImpulseNumerical;
clear LHLinearImpulseNumerical;
clear RHLinearImpulseNumerical;