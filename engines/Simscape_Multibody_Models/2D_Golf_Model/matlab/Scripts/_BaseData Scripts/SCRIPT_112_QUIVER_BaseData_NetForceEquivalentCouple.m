%Generate Club Quiver Plot
figure(112);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.TotalHandForceGlobal(:,1),BASEQ.TotalHandForceGlobal(:,2),BASEQ.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
NetForceQuiver.AutoScaleFactor=3;

%Generate Equivalent Couple Quiver Plot
NetCoupleQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),BASEQ.EquivalentMidpointCoupleGlobal(:,1),BASEQ.EquivalentMidpointCoupleGlobal(:,2),BASEQ.EquivalentMidpointCoupleGlobal(:,3));
NetCoupleQuiver.LineWidth=1;
NetCoupleQuiver.Color=[.8 .2 0];
NetCoupleQuiver.MaxHeadSize=0.1;
NetCoupleQuiver.AutoScaleFactor=3;


%Add Legend to Plot
legend('','','','','Net Force','Equivalent MP Couple');

%Add a Title
title('Net Force and Equivalent MP Couple');
subtitle('BASE');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('BaseData Quiver Plots/BASE_Quiver Plot - Net Force and Equivalent MP Couple');
pause(PauseTime);

%Close Figure
close(112);

%Clear Figure from Workspace
clear NetForceQuiver;
clear NetCoupleQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
