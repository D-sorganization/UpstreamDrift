%Generate Club Quiver Plot
figure(913);
run SCRIPT_QuiverClubandShaftData.m;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(Data.MPx(:,1),Data.MPy(:,1),Data.MPz(:,1),Data.TotalHandForceGlobal(:,1),Data.TotalHandForceGlobal(:,2),Data.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
NetForceQuiver.AutoScaleFactor=3;
%Add Legend to Plot
legend('','','','','Net Force');

%Add a Title
title('Net Force');
subtitle('Data');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('Data Quiver Plots/Quiver Plot - Net Force');
pause(PauseTime);

%Close Figure
close(913);

%Clear Figure from Workspace
clear NetForceQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;