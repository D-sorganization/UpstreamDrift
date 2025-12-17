%Generate Club Quiver Plot
figure(512);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),DELTAQ.TotalHandForceGlobal(:,1),DELTAQ.TotalHandForceGlobal(:,2),DELTAQ.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
NetForceQuiver.AutoScaleFactor=3;

%Generate Equivalent Couple Quiver Plot
NetCoupleQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),DELTAQ.EquivalentMidpointCoupleGlobal(:,1),DELTAQ.EquivalentMidpointCoupleGlobal(:,2),DELTAQ.EquivalentMidpointCoupleGlobal(:,3));
NetCoupleQuiver.LineWidth=1;
NetCoupleQuiver.Color=[.8 .2 0];
NetCoupleQuiver.MaxHeadSize=0.1;
NetCoupleQuiver.AutoScaleFactor=3;


%Add Legend to Plot
legend('','','','','Net Force','Equivalent MP Couple');

%Add a Title
title('Net Force and Equivalent MP Couple');
subtitle('DELTA');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Delta Quiver Plots/DELTA_Quiver Plot - Net Force and Equivalent MP Couple');
pause(PauseTime);


%Close Figure
close(512);

%Clear Figure from Workspace
clear NetForceQuiver;
clear NetCoupleQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;