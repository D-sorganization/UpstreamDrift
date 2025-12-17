%Calculate the Maximum Speeds and the Times They Occur

%Generate CHS Array
h=height(Data);
for i=1:h
CHSTemp=Data{i,["CHS (mph)"]};
CHS(i,1)=CHSTemp;
end

%Generate Hand Speed Array
for i=1:h
HSTemp=Data{i,["Hand Speed (mph)"]};
HS(i,1)=HSTemp;
end

%Find Max CHS Value
MaxCHS=max(CHS);
SummaryTable.("MaxCHS")=MaxCHS;

%Find Max Hand Speed Value
MaxHandSpeed=max(HS);
SummaryTable.("MaxHandSpeed")=MaxHandSpeed;

%Cleanup
clear i;
clear h
clear CHSTem;
clear HSTemp;

%Find the row in the table where each maximum occurs
CHSMaxRow=find(CHS==MaxCHS,1);
HSMaxRow=find(HS==MaxHandSpeed,1);

%Find the time in the table where the maximum occurs
CHSMaxTime=Data.Time(CHSMaxRow,1);
HandSpeedMaxTime=Data.Time(HSMaxRow,1);
SummaryTable.("HandSpeedMaxTime")=HandSpeedMaxTime;


%Find AoA at time of maximum CHS
AoAatMaxCHS=Data.AoA(CHSMaxRow,1);
SummaryTable.("AoAatMaxCHS")=AoAatMaxCHS;


%Calculate the time that the equivalent midpoint couple goes negative in
%late downswing
TimeofAlphaReversal=interp1(Data.EquivalentMidpointCoupleLocal(50:end,3),Data.Time(50:end,1),0.0,'linear');
SummaryTable.("TimeofAlphaReversal")=TimeofAlphaReversal;

%Generate a table of the times when the function of interest (f) crosses
%zero.
f=Data.AoA;
t=Data.Time;

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
CHSZeroAoA=interp1(Data.Time,Data.("CHS (mph)"),TimeofZeroAoA,'linear');
SummaryTable.("CHSZeroAoA")=CHSZeroAoA;

%Find Data Needed for Grip Quivers at Time of Max CHS
ClubQuiverMaxCHSData.("ButtxMaxCHS")=interp1(Data.Time,Data.("Buttx"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("ButtyMaxCHS")=interp1(Data.Time,Data.("Butty"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("ButtzMaxCHS")=interp1(Data.Time,Data.("Buttz"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("GripdxMaxCHS")=interp1(Data.Time,Data.("Gripdx"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("GripdyMaxCHS")=interp1(Data.Time,Data.("Gripdy"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("GripdzMaxCHS")=interp1(Data.Time,Data.("Gripdz"),CHSMaxTime,'linear');

%Find Data Needed for Shaft Quivers at Time of Max CHS
ClubQuiverMaxCHSData.("RWxMaxCHS")=interp1(Data.Time,Data.("RWx"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("RWyMaxCHS")=interp1(Data.Time,Data.("RWy"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("RWzMaxCHS")=interp1(Data.Time,Data.("RWz"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("ShaftdxMaxCHS")=interp1(Data.Time,Data.("Shaftdx"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("ShaftdyMaxCHS")=interp1(Data.Time,Data.("Shaftdy"),CHSMaxTime,'linear');
ClubQuiverMaxCHSData.("ShaftdzMaxCHS")=interp1(Data.Time,Data.("Shaftdz"),CHSMaxTime,'linear');

%Find Data Needed for Grip Quivers at Time of Alpha Reversal
ClubQuiverAlphaReversalData.("ButtxAlphaReversal")=interp1(Data.Time,Data.("Buttx"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("ButtyAlphaReversal")=interp1(Data.Time,Data.("Butty"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("ButtzAlphaReversal")=interp1(Data.Time,Data.("Buttz"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("GripdxAlphaReversal")=interp1(Data.Time,Data.("Gripdx"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("GripdyAlphaReversal")=interp1(Data.Time,Data.("Gripdy"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("GripdzAlphaReversal")=interp1(Data.Time,Data.("Gripdz"),TimeofAlphaReversal,'linear');

%Find Data Needed for Shaft Quivers at Time of Alpha Reversal
ClubQuiverAlphaReversalData.("RWxAlphaReversal")=interp1(Data.Time,Data.("RWx"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("RWyAlphaReversal")=interp1(Data.Time,Data.("RWy"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("RWzAlphaReversal")=interp1(Data.Time,Data.("RWz"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("ShaftdxAlphaReversal")=interp1(Data.Time,Data.("Shaftdx"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("ShaftdyAlphaReversal")=interp1(Data.Time,Data.("Shaftdy"),TimeofAlphaReversal,'linear');
ClubQuiverAlphaReversalData.("ShaftdzAlphaReversal")=interp1(Data.Time,Data.("Shaftdz"),TimeofAlphaReversal,'linear');

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















