%Calculate the Maximum Speeds and the Times They Occur

%Generate CHS Array
h=height(BASEQ);
for i=1:h
CHSTemp=BASEQ{i,["CHS (mph)"]};
CHS(i,1)=CHSTemp;
end

%Find Max CHS Value
MaxCHS=max(CHS);
SummaryTable.("MaxCHS")=MaxCHS;

%Generate Hand Speed Array
for i=1:h
HSTemp=BASEQ{i,["Hand Speed (mph)"]};
HS(i,1)=HSTemp;
end

%Find Max Hand Speed Value
MaxHandSpeed=max(HS);
SummaryTable.("MaxHandSpeed")=MaxHandSpeed;

%Generate HipAV Array
h=height(BASEQ);
for i=1:h
HipAVTemp=BASEQ{i,["BaseAV"]};
HipAV(i,1)=HipAVTemp;
end

%Find Max Hip AV Value
MaxHipAV=max(HipAV);
SummaryTable.("MaxHipAV")=MaxHipAV;

%Generate TorsoAV Array
h=height(BASEQ);
for i=1:h
TorsoAVTemp=BASEQ{i,["ChestAV"]};
TorsoAV(i,1)=TorsoAVTemp;
end

%Find Max Torso AV Value
MaxTorsoAV=max(TorsoAV);
SummaryTable.("MaxTorsoAV")=MaxTorsoAV;

%Generate LScapAV Array
h=height(BASEQ);
for i=1:h
LScapAVTemp=BASEQ{i,["LScapAV"]};
LScapAV(i,1)=LScapAVTemp;
end

%Find Max LScap AV Value
MaxLScapAV=max(LScapAV);
SummaryTable.("MaxLScapAV")=MaxLScapAV;

%Generate LUpperArmAV Array
h=height(BASEQ);
for i=1:h
LUpperArmAVTemp=BASEQ{i,["LUpperArmAV"]};
LUpperArmAV(i,1)=LUpperArmAVTemp;
end

%Find Max LUpperArm AV Value
MaxLUpperArmAV=max(LUpperArmAV);
SummaryTable.("MaxLUpperArmAV")=MaxLUpperArmAV;

%Generate LForearmAV Array
h=height(BASEQ);
for i=1:h
LForearmAVTemp=BASEQ{i,["LForearmAV"]};
LForearmAV(i,1)=LForearmAVTemp;
end

%Find Max LForearm AV Value
MaxLForearmAV=max(LForearmAV);
SummaryTable.("MaxLForearmAV")=MaxLForearmAV;

%Generate ClubAV Array
h=height(BASEQ);
for i=1:h
ClubAVTemp=BASEQ{i,["ClubhandleAV"]};
ClubAV(i,1)=ClubAVTemp;
end

%Find Max LForearm AV Value
MaxClubAV=max(ClubAV);
SummaryTable.("MaxClubAV")=MaxClubAV;

%Cleanup
clear i;
clear h
clear CHSTemp;
clear HSTemp;
clear HipAVTemp;
clear TorsoAVTemp;
clear LScapAVTemp;
clear LUpperArmAVTemp;
clear LForearmAVTemp;
clear ClubAVTemp;

%Find the row in the table where each maximum occurs
CHSMaxRow=find(CHS==MaxCHS,1);
HSMaxRow=find(HS==MaxHandSpeed,1);
HipAVMaxRow=find(HipAV==MaxHipAV,1);
TorsoAVMaxRow=find(TorsoAV==MaxTorsoAV,1);
LScapAVMaxRow=find(LScapAV==MaxLScapAV,1);
LUpperArmAVMaxRow=find(LUpperArmAV==MaxLUpperArmAV,1);
LForearmAVMaxRow=find(LForearmAV==MaxLForearmAV,1);
ClubAVMaxRow=find(ClubAV==MaxClubAV,1);

%Find the time in the table where the maximum occurs
CHSMaxTime=BASEQ.Time(CHSMaxRow,1);
SummaryTable.("CHSMaxTime")=CHSMaxTime;

HandSpeedMaxTime=BASEQ.Time(HSMaxRow,1);
SummaryTable.("HandSpeedMaxTime")=HandSpeedMaxTime;

HipAVMaxTime=BASEQ.Time(HipAVMaxRow,1);
SummaryTable.("HipAVMaxTime")=HipAVMaxTime;

TorsoAVMaxTime=BASEQ.Time(TorsoAVMaxRow,1);
SummaryTable.("TorsoAVMaxTime")=TorsoAVMaxTime;

LScapAVMaxTime=BASEQ.Time(LScapAVMaxRow,1);
SummaryTable.("LScapAVMaxTime")=LScapAVMaxTime;

LUpperArmAVMaxTime=BASEQ.Time(LUpperArmAVMaxRow,1);
SummaryTable.("LUpperArmAVMaxTime")=LUpperArmAVMaxTime;

LForearmAVMaxTime=BASEQ.Time(LForearmAVMaxRow,1);
SummaryTable.("LForearmAVMaxTime")=LForearmAVMaxTime;

ClubAVMaxTime=BASEQ.Time(ClubAVMaxRow,1);
SummaryTable.("ClubAVMaxTime")=ClubAVMaxTime;

%Find AoA at time of maximum CHS
AoAatMaxCHS=BASEQ.AoA(CHSMaxRow,1);
SummaryTable.("AoAatMaxCHS")=AoAatMaxCHS;


%Calculate the time that the equivalent midpoint couple goes negative in
%late downswing
TimeofAlphaReversal=interp1(BASE.EquivalentMidpointCoupleLocal(50:end,3),BASE.Time(50:end,1),0.0,'linear');
SummaryTable.("TimeofAlphaReversal")=TimeofAlphaReversal;

%Calculate the time that the ZTCF equivalent midpoint couple goes negative in
%late downswing. Currently I cut off the first 50 data points so I don't
%capture any startup effects.

TimeofZTCFAlphaReversal=interp1(ZTCF.EquivalentMidpointCoupleLocal(50:end,3),ZTCF.Time(50:end,1),0.0,'linear');
SummaryTable.("TimeofZTCFAlphaReversal")=TimeofZTCFAlphaReversal;

%Calculate the time that the ZTCF equivalent midpoint couple goes negative in
%late downswing. Currently I cut off the first 50 data points so I don't
%capture any startup effects.

TimeofDELTAAlphaReversal=interp1(DELTA.EquivalentMidpointCoupleLocal(50:end,3),DELTA.Time(50:end,1),0.0,'linear');
SummaryTable.("TimeofDELTAAlphaReversal")=TimeofDELTAAlphaReversal;

%Generate a table of the times when the function of interest (f) crosses
%zero.
f=BASE.AoA;
t=BASE.Time;

idx = find( f(2:end).*f(1:end-1)<0 );
t_zero = zeros(size(idx));
for i=1:numel(idx)
    j = idx(i);
    t_zero(i) = interp1( f(j:j+1), t(j:j+1), 0.0, 'linear' );
end

%Time of Zero AoA that Occurs Last
TimeofZeroAoA=max(t_zero);
SummaryTable.("TimeofZeroAoA")=TimeofZeroAoA;

%CHS at time of zero AoA
CHSZeroAoA=interp1(BASEQ.Time,BASEQ.("CHS (mph)"),TimeofZeroAoA,'linear');
SummaryTable.("CHSZeroAoA")=CHSZeroAoA;

%Find Data Needed for Grip Quivers at Time of Max CHS
ClubQuiverMaxCHS.("ButtxMaxCHS")=interp1(BASE.Time,BASE.("Buttx"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("ButtyMaxCHS")=interp1(BASE.Time,BASE.("Butty"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("ButtzMaxCHS")=interp1(BASE.Time,BASE.("Buttz"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("GripdxMaxCHS")=interp1(BASE.Time,BASE.("Gripdx"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("GripdyMaxCHS")=interp1(BASE.Time,BASE.("Gripdy"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("GripdzMaxCHS")=interp1(BASE.Time,BASE.("Gripdz"),CHSMaxTime,'linear');

%Find Data Needed for Shaft Quivers at Time of Max CHS
ClubQuiverMaxCHS.("RWxMaxCHS")=interp1(BASE.Time,BASE.("RWx"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("RWyMaxCHS")=interp1(BASE.Time,BASE.("RWy"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("RWzMaxCHS")=interp1(BASE.Time,BASE.("RWz"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("ShaftdxMaxCHS")=interp1(BASE.Time,BASE.("Shaftdx"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("ShaftdyMaxCHS")=interp1(BASE.Time,BASE.("Shaftdy"),CHSMaxTime,'linear');
ClubQuiverMaxCHS.("ShaftdzMaxCHS")=interp1(BASE.Time,BASE.("Shaftdz"),CHSMaxTime,'linear');

%Find Data Needed for Grip Quivers at Time of Alpha Reversal
ClubQuiverAlphaReversal.("ButtxAlphaReversal")=interp1(BASE.Time,BASE.("Buttx"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("ButtyAlphaReversal")=interp1(BASE.Time,BASE.("Butty"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("ButtzAlphaReversal")=interp1(BASE.Time,BASE.("Buttz"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("GripdxAlphaReversal")=interp1(BASE.Time,BASE.("Gripdx"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("GripdyAlphaReversal")=interp1(BASE.Time,BASE.("Gripdy"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("GripdzAlphaReversal")=interp1(BASE.Time,BASE.("Gripdz"),TimeofAlphaReversal,'linear');

%Find Data Needed for Shaft Quivers at Time of Alpha Reversal
ClubQuiverAlphaReversal.("RWxAlphaReversal")=interp1(BASE.Time,BASE.("RWx"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("RWyAlphaReversal")=interp1(BASE.Time,BASE.("RWy"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("RWzAlphaReversal")=interp1(BASE.Time,BASE.("RWz"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("ShaftdxAlphaReversal")=interp1(BASE.Time,BASE.("Shaftdx"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("ShaftdyAlphaReversal")=interp1(BASE.Time,BASE.("Shaftdy"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversal.("ShaftdzAlphaReversal")=interp1(BASE.Time,BASE.("Shaftdz"),TimeofAlphaReversal,'linear');

%Find Data Needed for Grip Quivers at Time of ZTCF Alpha Reversal
ClubQuiverZTCFAlphaReversal.("ButtxZTCFAlphaReversal")=interp1(BASE.Time,BASE.("Buttx"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("ButtyZTCFAlphaReversal")=interp1(BASE.Time,BASE.("Butty"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("ButtzZTCFAlphaReversal")=interp1(BASE.Time,BASE.("Buttz"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("GripdxZTCFAlphaReversal")=interp1(BASE.Time,BASE.("Gripdx"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("GripdyZTCFAlphaReversal")=interp1(BASE.Time,BASE.("Gripdy"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("GripdzZTCFAlphaReversal")=interp1(BASE.Time,BASE.("Gripdz"),TimeofZTCFAlphaReversal,'linear');

%Find Data Needed for Shaft Quivers at Time of ZTCF Alpha Reversal
ClubQuiverZTCFAlphaReversal.("RWxZTCFAlphaReversal")=interp1(BASE.Time,BASE.("RWx"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("RWyZTCFAlphaReversal")=interp1(BASE.Time,BASE.("RWy"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("RWzZTCFAlphaReversal")=interp1(BASE.Time,BASE.("RWz"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("ShaftdxZTCFAlphaReversal")=interp1(BASE.Time,BASE.("Shaftdx"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("ShaftdyZTCFAlphaReversal")=interp1(BASE.Time,BASE.("Shaftdy"),TimeofZTCFAlphaReversal,'linear');
ClubQuiverZTCFAlphaReversal.("ShaftdzZTCFAlphaReversal")=interp1(BASE.Time,BASE.("Shaftdz"),TimeofZTCFAlphaReversal,'linear');

%Find Data Needed for Grip Quivers at Time of DELTA Alpha Reversal
ClubQuiverDELTAAlphaReversal.("ButtxDELTAAlphaReversal")=interp1(BASE.Time,BASE.("Buttx"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("ButtyDELTAAlphaReversal")=interp1(BASE.Time,BASE.("Butty"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("ButtzDELTAAlphaReversal")=interp1(BASE.Time,BASE.("Buttz"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("GripdxDELTAAlphaReversal")=interp1(BASE.Time,BASE.("Gripdx"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("GripdyDELTAAlphaReversal")=interp1(BASE.Time,BASE.("Gripdy"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("GripdzDELTAAlphaReversal")=interp1(BASE.Time,BASE.("Gripdz"),TimeofDELTAAlphaReversal,'linear');

%Find Data Needed for Shaft Quivers at Time of DELTA Alpha Reversal
ClubQuiverDELTAAlphaReversal.("RWxDELTAAlphaReversal")=interp1(BASE.Time,BASE.("RWx"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("RWyDELTAAlphaReversal")=interp1(BASE.Time,BASE.("RWy"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("RWzDELTAAlphaReversal")=interp1(BASE.Time,BASE.("RWz"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("ShaftdxDELTAAlphaReversal")=interp1(BASE.Time,BASE.("Shaftdx"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("ShaftdyDELTAAlphaReversal")=interp1(BASE.Time,BASE.("Shaftdy"),TimeofDELTAAlphaReversal,'linear');
ClubQuiverDELTAAlphaReversal.("ShaftdzDELTAAlphaReversal")=interp1(BASE.Time,BASE.("Shaftdz"),TimeofDELTAAlphaReversal,'linear');


clear CHS;
clear HS;
clear i;
clear j;
clear idx;
clear CHSMaxRow;
clear CHSMaxTime;
clear HSMaxRow;
clear HSMaxTime;
clear CHSTemp;
clear CHSZeroAoA;
clear f;
clear HandSpeedMaxTime;
clear MaxCHS;
clear MaxHandSpeed;
clear t;
clear t_zero;
clear TimeofAlphaReversal;
clear TimeofZeroAoA;
clear AoAatMaxCHS;
clear ClubAV;
clear ClubAVMaxRow;
clear ClubAVMaxTime;
clear HipAV;
clear HipAVMaxRow;
clear HipAVMaxTime;
clear LForearmAV;
clear LForearmAVMaxRow;
clear LForearmAVMaxTime;
clear LScapAV;
clear LScapAVMaxRow;
clear LScapAVMaxTime;
clear LUpperArmAV;
clear LUpperArmAVMaxRow;
clear LUpperArmAVMaxTime
clear MaxClubAV;
clear MaxHipAV;
clear MaxLForearmAV;
clear MaxLScapAV;
clear MaxLUpperArmAV;
clear MaxTorsoAV;
clear TorsoAV;
clear TorsoAVMaxRow;
clear TorsoAVMaxTime;
clear TimeofZTCFAlphaReversal;
clear TimeofDELTAAlphaReversal;















