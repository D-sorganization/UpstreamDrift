%Generate Club Quiver Plot
figure(722);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Generate Left Hand Sum of Moments Quiver Plot
LWSumofMomentsQuiver=quiver3(BASEQ.LWx(:,1),BASEQ.LWy(:,1),BASEQ.LWz(:,1),BASEQ.SumofMomentsLWristonClub(:,1),BASEQ.SumofMomentsLWristonClub(:,2),BASEQ.SumofMomentsLWristonClub(:,3));
LWSumofMomentsQuiver.LineWidth=1;
LWSumofMomentsQuiver.Color=[0 1 0];
LWSumofMomentsQuiver.MaxHeadSize=0.1;
LWSumofMomentsQuiver.AutoScaleFactor=3;

%Generate Right Hand Sum of Moments Quiver Plot
RWSumofMomentsQuiver=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.SumofMomentsRWristonClub(:,1),BASEQ.SumofMomentsRWristonClub(:,2),BASEQ.SumofMomentsRWristonClub(:,3));
RWSumofMomentsQuiver.LineWidth=1;
RWSumofMomentsQuiver.Color=[.8 .2 0];
RWSumofMomentsQuiver.MaxHeadSize=0.1;
%Correct scaling on Sum of Momentss so that ZTCF and BASE are scaled the same.
RWSumofMomentsQuiver.AutoScaleFactor=LWSumofMomentsQuiver.ScaleFactor/RWSumofMomentsQuiver.ScaleFactor;

%Generate ZTCF Left Hand Sum of Moments Quiver Plot
ZTCFLWSumofMomentsQuiver=quiver3(ZTCFQ.LWx(:,1),ZTCFQ.LWy(:,1),ZTCFQ.LWz(:,1),ZTCFQ.SumofMomentsLWristonClub(:,1),ZTCFQ.SumofMomentsLWristonClub(:,2),ZTCFQ.SumofMomentsLWristonClub(:,3));
ZTCFLWSumofMomentsQuiver.LineWidth=1;
ZTCFLWSumofMomentsQuiver.Color=[0 0.3 0];
ZTCFLWSumofMomentsQuiver.MaxHeadSize=0.1;
%Correct scaling on Sum of Momentss so that ZTCF and BASE are scaled the same.
ZTCFLWSumofMomentsQuiver.AutoScaleFactor=LWSumofMomentsQuiver.ScaleFactor/ZTCFLWSumofMomentsQuiver.ScaleFactor;

%Generate ZTCF Right Hand Sum of Moments Quiver Plot
ZTCFRWSumofMomentsQuiver=quiver3(ZTCFQ.RWx(:,1),ZTCFQ.RWy(:,1),ZTCFQ.RWz(:,1),ZTCFQ.SumofMomentsRWristonClub(:,1),ZTCFQ.SumofMomentsRWristonClub(:,2),ZTCFQ.SumofMomentsRWristonClub(:,3));
ZTCFRWSumofMomentsQuiver.LineWidth=1;
ZTCFRWSumofMomentsQuiver.Color=[.5 .5 0];
ZTCFRWSumofMomentsQuiver.MaxHeadSize=0.1;
%Correct scaling on Sum of Moments so that ZTCF and BASE are scaled the same.
ZTCFRWSumofMomentsQuiver.AutoScaleFactor=LWSumofMomentsQuiver.ScaleFactor/ZTCFRWSumofMomentsQuiver.ScaleFactor;



%Add Legend to Plot
legend('','','','','BASE - LW Sum of Moments','BASE - RW Sum of Moments','ZTCF - LW Sum of Moments','ZTCF - RW Sum of Moments');

%Add a Title
title('LW and RW Sum of Moments on Club');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - LWRW Sum of Moments on Club');
pause(PauseTime);

%Close Figure
close(722);

%Clear Figure from Workspace
clear LWSumofMomentsQuiver;
clear RWSumofMomentsQuiver;
clear ZTCFLWSumofMomentsQuiver;
clear ZTCFRWSumofMomentsQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
