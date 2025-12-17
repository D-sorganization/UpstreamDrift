%Generate Club Quiver Plot
figure(723);
run SCRIPT_QuiverClubandShaftBaseData.m;

%Create Total Sum of Moments Array by Adding LH and RH Sum of Moments
TotalSumofMoments=BASEQ.SumofMomentsLWristonClub+BASEQ.SumofMomentsRWristonClub;
ZTCFTotalSumofMoments=ZTCFQ.SumofMomentsLWristonClub+ZTCFQ.SumofMomentsRWristonClub;

%Generate Total Hand Sum of Moments Quiver Plot
SumofMomentsQuiver=quiver3(BASEQ.MPx(:,1),BASEQ.MPy(:,1),BASEQ.MPz(:,1),TotalSumofMoments(:,1),TotalSumofMoments(:,2),TotalSumofMoments(:,3));
SumofMomentsQuiver.LineWidth=1;
SumofMomentsQuiver.Color=[0 1 0];
SumofMomentsQuiver.MaxHeadSize=0.1;
SumofMomentsQuiver.AutoScaleFactor=3;

%Generate ZTCF Total Hand Sum of Moments Quiver Plot
ZTCFSumofMomentsQuiver=quiver3(ZTCFQ.MPx(:,1),ZTCFQ.MPy(:,1),ZTCFQ.MPz(:,1),ZTCFTotalSumofMoments(:,1),ZTCFTotalSumofMoments(:,2),ZTCFTotalSumofMoments(:,3));
ZTCFSumofMomentsQuiver.LineWidth=1;
ZTCFSumofMomentsQuiver.Color=[0 0.3 0];
ZTCFSumofMomentsQuiver.MaxHeadSize=0.1;
%Correct scaling on Sum of Momentss so that ZTCF and BASE are scaled the same.
ZTCFSumofMomentsQuiver.AutoScaleFactor=SumofMomentsQuiver.ScaleFactor/ZTCFSumofMomentsQuiver.ScaleFactor;

%Add Legend to Plot
legend('','','','','BASE - Total Sum of Moments','ZTCF - Total Sum of Moments');

%Add a Title
title('Total Sum of Moments on Club');
subtitle('COMPARISON');

%Set View
view(-0.0885,-10.6789);

%Save Figure
savefig('Comparison Quiver Plots/COMPARISON_Quiver Plot - Total Sum of Moments on Club');
pause(PauseTime);

%Close Figure
close(723);

%Clear Figure from Workspace
clear SumofMomentsQuiver;
clear RWSumofMomentsQuiver;
clear ZTCFSumofMomentsQuiver;
clear ZTCFRWSumofMomentsQuiver;
clear Grip;
clear Shaft;
clear ClubPath;
clear HandPath;
clear TotalSumofMoments;
clear ZTCFTotalSumofMoments;