%Generate Club Quiver Plot
figure(147);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Total Force Quiver Plot
NetForceQuiver=quiver3(BASEQ.HipGlobalPosition(:,1),BASEQ.HipGlobalPosition(:,2),BASEQ.HipGlobalPosition(:,3),BASEQ.BaseonHipForceGlobal(:,1),BASEQ.BaseonHipForceGlobal(:,2),BASEQ.BaseonHipForceGlobal(:,3));
NetForceQuiver.LineWidth=1;
NetForceQuiver.Color=[0 1 0];
NetForceQuiver.MaxHeadSize=0.1;
NetForceQuiver.AutoScaleFactor=3;
%Add Legend to Plot
legend('','','','','Base on Hip Force');

%Add a Title
title('Base on Hip Force');
subtitle('BASE');

%Set View
view(-0.186585735654603,37.199999973925109);

%Save Figure
savefig('BaseData Quiver Plots//BASE_Quiver Plot - Base on Hip Force');
pause(PauseTime);

%Close Figure
close(147);

%Clear Figure from Workspace
clear NetForceQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;