%Generate Club Quiver Plot
figure(912);
run SCRIPT_QuiverClubandShaftData.m;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(Data.MPx(:,1),Data.MPy(:,1),Data.MPz(:,1),Data.TotalHandForceGlobal(:,1),Data.TotalHandForceGlobal(:,2),Data.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
NetForceQuiver.AutoScaleFactor=3;

%Generate Equivalent Couple Quiver Plot
NetCoupleQuiver=quiver3(Data.MPx(:,1),Data.MPy(:,1),Data.MPz(:,1),Data.EquivalentMidpointCoupleGlobal(:,1),Data.EquivalentMidpointCoupleGlobal(:,2),Data.EquivalentMidpointCoupleGlobal(:,3));
NetCoupleQuiver.LineWidth=1;
NetCoupleQuiver.Color=[.8 .2 0];
NetCoupleQuiver.MaxHeadSize=0.1;
NetCoupleQuiver.AutoScaleFactor=3;


%Add Legend to Plot
legend('','','','','Net Force','Equivalent MP Couple');

%Add a Title
title('Net Force and Equivalent MP Couple');
subtitle('Data');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Data Quiver Plots/Quiver Plot - Net Force and Equivalent MP Couple');
pause(PauseTime);

%Close Figure
close(912);

%Clear Figure from Workspace
clear NetForceQuiver;
clear NetCoupleQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;