%Generate Club Quiver Plot
figure(313);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFQ.TotalHandForceGlobal(:,1),ZTCFQ.TotalHandForceGlobal(:,2),ZTCFQ.TotalHandForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
NetForceQuiver.AutoScaleFactor=3;
%Add Legend to Plot
legend('','','','','Net Force');

%Add a Title
title('Net Force');
subtitle('ZTCF');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('ZTCF Quiver Plots/ZTCF_Quiver Plot - Net Force');
pause(PauseTime);

%Close Figure
close(313);

%Clear Figure from Workspace
clear NetForceQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;