function PLOT_Comparison_Plots()
%%% ====== Start of SCRIPT_704_3D_PLOT_Comparison_ForceAlongHandPath.m ======
figure(704);
hold on;
plot(BASEQ.Time,BASEQ.ForceAlongHandPath,'LineWidth',3);
plot(ZTCFQ.Time,ZTCFQ.ForceAlongHandPath,'--','LineWidth',3);
plot(DELTAQ.Time,DELTAQ.ForceAlongHandPath,':','LineWidth',3);
xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('BASE','ZTCF','DELTA');
legend('Location','southeast');
%Add a Title
title('Force Along Hand Path');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_Plot - Force Along Hand Path');
pause(PauseTime);
%Close Figure
close(704);

%% ====== End of SCRIPT_704_3D_PLOT_Comparison_ForceAlongHandPath.m ======

%% ====== Start of SCRIPT_705_3D_PLOT_Comparison_LinearWork.m ======
figure(705);
hold on;
plot(BASEQ.Time,BASEQ.LSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.RSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.LELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.RELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.LSLinearWorkonArm,'--');
plot(ZTCFQ.Time,ZTCFQ.RSLinearWorkonArm,'--');
plot(ZTCFQ.Time,ZTCFQ.LELinearWorkonForearm,'--');
plot(ZTCFQ.Time,ZTCFQ.RELinearWorkonForearm,'--');
plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub,'--');
plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub,'--');
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Work - BASE','RS Linear Work - BASE','LE Linear Work - BASE','RE Linear Work - BASE','LW Linear Work - BASE','RW Linear Work - BASE','LS Linear Work - ZTCF','RS Linear Work - ZTCF','LE Linear Work - ZTCF','RE Linear Work - ZTCF','LW Linear Work - ZTCF','RW Linear Work - ZTCF');
legend('Location','southeast');
%Add a Title
title('Linear Work on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Linear Work on Distal');
pause(PauseTime);
%Close Figure
close(705);

%% ====== End of SCRIPT_705_3D_PLOT_Comparison_LinearWork.m ======

%% ====== Start of SCRIPT_706_3D_PLOT_Comparison_AngularWork.m ======
figure(706);
hold on;
plot(BASEQ.Time,BASEQ.LSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.RSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.LEAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.REAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.LHAngularWorkonClub);
plot(BASEQ.Time,BASEQ.RHAngularWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.LSAngularWorkonArm,'--');
plot(ZTCFQ.Time,ZTCFQ.RSAngularWorkonArm,'--');
plot(ZTCFQ.Time,ZTCFQ.LEAngularWorkonForearm,'--');
plot(ZTCFQ.Time,ZTCFQ.REAngularWorkonForearm,'--');
plot(ZTCFQ.Time,ZTCFQ.LHAngularWorkonClub,'--');
plot(ZTCFQ.Time,ZTCFQ.RHAngularWorkonClub,'--');
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Work - BASE','RS Angular Work - BASE','LE Angular Work - BASE','RE Angular Work - BASE','LH Angular Work - BASE','RH Angular Work - BASE','LS Angular Work - ZTCF','RS Angular Work - ZTCF','LE Angular Work - ZTCF','RE Angular Work - ZTCF','LH Angular Work - ZTCF','RH Angular Work - ZTCF');
legend('Location','southeast');
%Add a Title
title('Angular Work on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Angular Work on Distal');
pause(PauseTime);
%Close Figure
close(706);

%% ====== End of SCRIPT_706_3D_PLOT_Comparison_AngularWork.m ======

%% ====== Start of SCRIPT_707_3D_PLOT_Comparison_TotalWork.m ======
figure(707);
hold on;
plot(BASEQ.Time,BASEQ.TotalLSWork);
plot(BASEQ.Time,BASEQ.TotalRSWork);
plot(BASEQ.Time,BASEQ.TotalLEWork);
plot(BASEQ.Time,BASEQ.TotalREWork);
plot(BASEQ.Time,BASEQ.TotalLHWork);
plot(BASEQ.Time,BASEQ.TotalRHWork);
plot(ZTCFQ.Time,ZTCFQ.TotalLSWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRSWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalLEWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalREWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalLHWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRHWork,'--');
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Total Work - BASE','RS Total Work - BASE','LE Total Work - BASE','RE Total Work - BASE','LH Total Work - BASE','RH Total Work - BASE','LS Total Work - ZTCF','RS Total Work - ZTCF','LE Total Work - ZTCF','RE Total Work - ZTCF','LH Total Work - ZTCF','RH Total Work - ZTCF');
legend('Location','southeast');
%Add a Title
title('Total Work on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total Work on Distal');
pause(PauseTime);
%Close Figure
close(707);

%% ====== End of SCRIPT_707_3D_PLOT_Comparison_TotalWork.m ======

%% ====== Start of SCRIPT_708_3D_PLOT_Comparison_LinearPower.m ======
figure(708);
hold on;
plot(BASEQ.Time,BASEQ.LSonArmLinearPower);
plot(BASEQ.Time,BASEQ.RSonArmLinearPower);
plot(BASEQ.Time,BASEQ.LEonForearmLinearPower);
plot(BASEQ.Time,BASEQ.REonForearmLinearPower);
plot(BASEQ.Time,BASEQ.LHonClubLinearPower);
plot(BASEQ.Time,BASEQ.RHonClubLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LSonArmLinearPower,'--');
plot(ZTCFQ.Time,ZTCFQ.RSonArmLinearPower,'--');
plot(ZTCFQ.Time,ZTCFQ.LEonForearmLinearPower,'--');
plot(ZTCFQ.Time,ZTCFQ.REonForearmLinearPower,'--');
plot(ZTCFQ.Time,ZTCFQ.LHonClubLinearPower,'--');
plot(ZTCFQ.Time,ZTCFQ.RHonClubLinearPower,'--');
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Power - BASE','RS Linear Power - BASE','LE Linear Power - BASE','RE Linear Power - BASE','LH Linear Power - BASE','RH Linear Power - BASE','LS Linear Power - ZTCF','RS Linear Power - ZTCF','LE Linear Power - ZTCF','RE Linear Power - ZTCF','LH Linear Power - ZTCF','RH Linear Power - ZTCF');
legend('Location','southeast');
%Add a Title
title('Linear Power on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Linear Power on Distal');
pause(PauseTime);
%Close Figure
close(708);

%% ====== End of SCRIPT_708_3D_PLOT_Comparison_LinearPower.m ======

%% ====== Start of SCRIPT_709_3D_PLOT_Comparison_AngularPower.m ======
figure(709);
hold on;
plot(BASEQ.Time,BASEQ.LSonArmAngularPower);
plot(BASEQ.Time,BASEQ.RSonArmAngularPower);
plot(BASEQ.Time,BASEQ.LEonForearmAngularPower);
plot(BASEQ.Time,BASEQ.REonForearmAngularPower);
plot(BASEQ.Time,BASEQ.LHonClubAngularPower);
plot(BASEQ.Time,BASEQ.RHonClubAngularPower);
plot(ZTCFQ.Time,ZTCFQ.LSonArmAngularPower,'--');
plot(ZTCFQ.Time,ZTCFQ.RSonArmAngularPower,'--');
plot(ZTCFQ.Time,ZTCFQ.LEonForearmAngularPower,'--');
plot(ZTCFQ.Time,ZTCFQ.REonForearmAngularPower,'--');
plot(ZTCFQ.Time,ZTCFQ.LHonClubAngularPower,'--');
plot(ZTCFQ.Time,ZTCFQ.RHonClubAngularPower,'--');
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Power - BASE','RS Angular Power - BASE','LE Angular Power - BASE','RE Angular Power - BASE','LH Angular Power - BASE','RH Angular Power - BASE','LS Angular Power - ZTCF','RS Angular Power - ZTCF','LE Angular Power - ZTCF','RE Angular Power - ZTCF','LH Angular Power - ZTCF','RH Angular Power - ZTCF');
legend('Location','southeast');
%Add a Title
title('Angular Power on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Angular Power on Distal');
pause(PauseTime);
%Close Figure
close(709);

%% ====== End of SCRIPT_709_3D_PLOT_Comparison_AngularPower.m ======

%% ====== Start of SCRIPT_710_3D_PLOT_Comparison_TotalPower.m ======
figure(710);
hold on;
plot(BASEQ.Time,BASEQ.TotalLSPower);
plot(BASEQ.Time,BASEQ.TotalRSPower);
plot(BASEQ.Time,BASEQ.TotalLEPower);
plot(BASEQ.Time,BASEQ.TotalREPower);
plot(BASEQ.Time,BASEQ.TotalLHPower);
plot(BASEQ.Time,BASEQ.TotalRHPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLSPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRSPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalLEPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalREPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalLHPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRHPower,'--');
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Total Power - BASE','RS Total Power - BASE','LE Total Power - BASE','RE Total Power - BASE','LH Total Power - BASE','RH Total Power - BASE','LS Total Power - ZTCF','RS Total Power - ZTCF','LE Total Power - ZTCF','RE Total Power - ZTCF','LH Total Power - ZTCF','RH Total Power - ZTCF');
legend('Location','southeast');
%Add a Title
title('Total Power on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total Power on Distal');
pause(PauseTime);
%Close Figure
close(710);

%% ====== End of SCRIPT_710_3D_PLOT_Comparison_TotalPower.m ======

%% ====== Start of SCRIPT_711_3D_PLOT_Comparison_LinearWorkClubOnly.m ======
figure(711);
hold on;
plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub,'--');
plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub,'--');
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Work - BASE','RH Linear Work - BASE','LH Linear Work - ZTCF','RH Linear Work - ZTCF');
legend('Location','southeast');
%Add a Title
title('Linear Work on Club');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Linear Work on Club');
pause(PauseTime);
%Close Figure
close(711);

%% ====== End of SCRIPT_711_3D_PLOT_Comparison_LinearWorkClubOnly.m ======

%% ====== Start of SCRIPT_712_3D_PLOT_Comparison_AngularWorkClubOnly.m ======
figure(712);
hold on;
plot(BASEQ.Time,BASEQ.LHAngularWorkonClub);
plot(BASEQ.Time,BASEQ.RHAngularWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.LHAngularWorkonClub,'--');
plot(ZTCFQ.Time,ZTCFQ.RHAngularWorkonClub,'--');
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Angular Work - BASE','RH Angular Work - BASE','LH Angular Work - ZTCF','RH Angular Work - ZTCF');
legend('Location','southeast');
%Add a Title
title('Angular Work on Club');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Angular Work on Club');
pause(PauseTime);
%Close Figure
close(712);

%% ====== End of SCRIPT_712_3D_PLOT_Comparison_AngularWorkClubOnly.m ======

%% ====== Start of SCRIPT_713_3D_PLOT_Comparison_TotalWorkClubOnly.m ======
figure(713);
hold on;
plot(BASEQ.Time,BASEQ.TotalLHWork);
plot(BASEQ.Time,BASEQ.TotalRHWork);
plot(ZTCFQ.Time,ZTCFQ.TotalLHWork,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRHWork,'--');
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Total Work - BASE','RH Total Work - BASE','LH Total Work - ZTCF','RH Total Work - ZTCF');
legend('Location','southeast');
%Add a Title
title('Total Work Club');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total Work on Club');
pause(PauseTime);
%Close Figure
close(713);

%% ====== End of SCRIPT_713_3D_PLOT_Comparison_TotalWorkClubOnly.m ======

%% ====== Start of SCRIPT_714_3D_PLOT_Comparison_LinearPowerClubOnly.m ======
figure(714);
hold on;
plot(BASEQ.Time,BASEQ.LHonClubLinearPower);
plot(BASEQ.Time,BASEQ.RHonClubLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LHonClubLinearPower,'--');
plot(ZTCFQ.Time,ZTCFQ.RHonClubLinearPower,'--');
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Power - BASE','RH Linear Power - BASE','LH Linear Power - ZTCF','RH Linear Power - ZTCF');
legend('Location','southeast');
%Add a Title
title('Linear Power on Club');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Linear Power on Club');
pause(PauseTime);
%Close Figure
close(714);

%% ====== End of SCRIPT_714_3D_PLOT_Comparison_LinearPowerClubOnly.m ======

%% ====== Start of SCRIPT_715_3D_PLOT_Comparison_AngularPowerClubOnly.m ======
figure(715);
hold on;
plot(BASEQ.Time,BASEQ.LHonClubAngularPower);
plot(BASEQ.Time,BASEQ.RHonClubAngularPower);
plot(ZTCFQ.Time,ZTCFQ.LHonClubAngularPower,'--');
plot(ZTCFQ.Time,ZTCFQ.RHonClubAngularPower,'--');
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LH Angular Power - BASE','RH Angular Power - BASE','LH Angular Power - ZTCF','RH Angular Power - ZTCF');
legend('Location','southeast');
%Add a Title
title('Angular Power on Club');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Angular Power on Club');
pause(PauseTime);
%Close Figure
close(715);

%% ====== End of SCRIPT_715_3D_PLOT_Comparison_AngularPowerClubOnly.m ======

%% ====== Start of SCRIPT_716_3D_PLOT_Comparison_TotalPowerClubOnly.m ======
figure(716);
hold on;
plot(BASEQ.Time,BASEQ.TotalLHPower);
plot(BASEQ.Time,BASEQ.TotalRHPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLHPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRHPower,'--');
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LH Total Power - BASE','RH Total Power - BASE','LH Total Power - ZTCF','RH Total Power - ZTCF');
legend('Location','southeast');
%Add a Title
title('Total Power on Club');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total Power on Club');
pause(PauseTime);
%Close Figure
close(716);

%% ====== End of SCRIPT_716_3D_PLOT_Comparison_TotalPowerClubOnly.m ======

%% ====== Start of SCRIPT_726_3D_PLOT_Comparison_ZTCFFractionalWork.m ======
figure(726);
hold on;
plot(BASEQ.Time,BASEQ.ZTCFQLSFractionalWork);
plot(BASEQ.Time,BASEQ.ZTCFQRSFractionalWork);
plot(BASEQ.Time,BASEQ.ZTCFQLEFractionalWork);
plot(BASEQ.Time,BASEQ.ZTCFQREFractionalWork);
plot(BASEQ.Time,BASEQ.ZTCFQLHFractionalWork);
plot(BASEQ.Time,BASEQ.ZTCFQRHFractionalWork);
ylim([-5 5]);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Total ZTCF Fractional Work','RS Total ZTCF Fractional Work','LE Total ZTCF Fractional Work','RE Total ZTCF Fractional Work','LH Total ZTCF Fractional Work','RH Total ZTCF Fractional Work');
legend('Location','southeast');
%Add a Title
title('Total ZTCF Fractional Work on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total ZTCF Fractional Work');
pause(PauseTime);
%Close Figure
close(726);

%% ====== End of SCRIPT_726_3D_PLOT_Comparison_ZTCFFractionalWork.m ======

%% ====== Start of SCRIPT_727_3D_PLOT_Comparison_DELTAFractionalWork.m ======
figure(727);
hold on;
plot(BASEQ.Time,BASEQ.DELTAQLSFractionalWork);
plot(BASEQ.Time,BASEQ.DELTAQRSFractionalWork);
plot(BASEQ.Time,BASEQ.DELTAQLEFractionalWork);
plot(BASEQ.Time,BASEQ.DELTAQREFractionalWork);
plot(BASEQ.Time,BASEQ.DELTAQLHFractionalWork);
plot(BASEQ.Time,BASEQ.DELTAQRHFractionalWork);
ylim([-5 5]);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Total DELTA Fractional Work','RS Total DELTA Fractional Work','LE Total DELTA Fractional Work','RE Total DELTA Fractional Work','LH Total DELTA Fractional Work','RH Total DELTA Fractional Work');
legend('Location','southeast');
%Add a Title
title('Total DELTA Fractional Work on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total DELTA Fractional Work');
pause(PauseTime);
%Close Figure
close(727);

%% ====== End of SCRIPT_727_3D_PLOT_Comparison_DELTAFractionalWork.m ======

%% ====== Start of SCRIPT_728_3D_PLOT_Comparison_ZTCFFractionalPower.m ======
figure(728);
hold on;
plot(BASEQ.Time,BASEQ.ZTCFQLSFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQRSFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQLEFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQREFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQLHFractionalPower);
plot(BASEQ.Time,BASEQ.ZTCFQRHFractionalPower);
ylim([-5 5]);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Total ZTCF Fractional Power','RS Total ZTCF Fractional Power','LE Total ZTCF Fractional Power','RE Total ZTCF Fractional Power','LW Total ZTCF Fractional Power','RH Total ZTCF Fractional Power');
legend('Location','southeast');
%Add a Title
title('Total ZTCF Fractional Power on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total ZTCF Fractional Power');
pause(PauseTime);
%Close Figure
close(728);

%% ====== End of SCRIPT_728_3D_PLOT_Comparison_ZTCFFractionalPower.m ======

%% ====== Start of SCRIPT_729_3D_PLOT_Comparison_DeltaFractionalPower.m ======
figure(729);
hold on;
plot(BASEQ.Time,BASEQ.DELTAQLSFractionalPower);
plot(BASEQ.Time,BASEQ.DELTAQRSFractionalPower);
plot(BASEQ.Time,BASEQ.DELTAQLEFractionalPower);
plot(BASEQ.Time,BASEQ.DELTAQREFractionalPower);
plot(BASEQ.Time,BASEQ.DELTAQLHFractionalPower);
plot(BASEQ.Time,BASEQ.DELTAQRHFractionalPower);
ylim([-5 5]);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Total DELTA Fractional Power','RS Total DELTA Fractional Power','LE Total DELTA Fractional Power','RE Total DELTA Fractional Power','LH Total DELTA Fractional Power','RH Total DELTA Fractional Power');
legend('Location','southeast');
%Add a Title
title('Total DELTA Fractional Power on Distal Segment');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total DELTA Fractional Power');
pause(PauseTime);
%Close Figure
close(729);

%% ====== End of SCRIPT_729_3D_PLOT_Comparison_DeltaFractionalPower.m ======

%% ====== Start of SCRIPT_748_3D_PLOT_Comparison_LocalHandForces.m ======
figure(748);
hold on;
plot(BASEQ.Time(:,1),BASEQ.LHonClubForceLocal(:,1),'LineWidth',3);
plot(BASEQ.Time(:,1),BASEQ.LHonClubForceLocal(:,2),'LineWidth',3);
plot(BASEQ.Time(:,1),BASEQ.LHonClubForceLocal(:,3),'LineWidth',3);
plot(BASEQ.Time(:,1),BASEQ.RHonClubForceLocal(:,1),'LineWidth',3);
plot(BASEQ.Time(:,1),BASEQ.RHonClubForceLocal(:,2),'LineWidth',3);
plot(BASEQ.Time(:,1),BASEQ.RHonClubForceLocal(:,3),'LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.LHonClubForceLocal(:,1),'LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.LHonClubForceLocal(:,2),'LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.LHonClubForceLocal(:,3),'LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.RHonClubForceLocal(:,1),'LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.RHonClubForceLocal(:,2),'LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.RHonClubForceLocal(:,3),'LineWidth',3);
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('BASE Left Wrist X','BASE Left Wrist Y','BASE Left Wrist Z',...
'BASE Right Wrist X','BASE Right Wrist Y','BASE Right Wrist Z',...
'ZTCF Left Wrist X','ZTCF Left Wrist Y','ZTCF Left Wrist Z',...
'ZTCF Right Wrist X','ZTCF Right Wrist Y','ZTCF Right Wrist Z');
legend('Location','southeast');
%Add a Title
title('Local Hand Forces on Club');
subtitle('COMPARISON');
%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Local Hand Forces');
pause(PauseTime);
%Close Figure
close(748);

%% ====== End of SCRIPT_748_3D_PLOT_Comparison_LocalHandForces.m ======

end
