function PLOT_DELTA_Plots()
%%% ====== Start of SCRIPT_501_3D_PLOT_DELTA_AngularWork.m ======
figure(501);
hold on;
plot(DELTAQ.Time,DELTAQ.LSAngularWorkonArm);
plot(DELTAQ.Time,DELTAQ.RSAngularWorkonArm);
plot(DELTAQ.Time,DELTAQ.LEAngularWorkonForearm);
plot(DELTAQ.Time,DELTAQ.REAngularWorkonForearm);
plot(DELTAQ.Time,DELTAQ.LHAngularWorkonClub);
plot(DELTAQ.Time,DELTAQ.RHAngularWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Work','RS Angular Work','LE Angular Work','RE Angular Work','LH Angular Work','RH Angular Work');
legend('Location','southeast');
%Add a Title
title('Angular Work on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Angular Work');
pause(PauseTime);
%Close Figure
close(501);

%% ====== End of SCRIPT_501_3D_PLOT_DELTA_AngularWork.m ======

%% ====== Start of SCRIPT_502_3D_PLOT_DELTA_AngularPower.m ======
figure(502);
hold on;
plot(ZTCFQ.Time,DELTAQ.LSonArmAngularPower);
plot(ZTCFQ.Time,DELTAQ.RSonArmAngularPower);
plot(ZTCFQ.Time,DELTAQ.LEonForearmAngularPower);
plot(ZTCFQ.Time,DELTAQ.REonForearmAngularPower);
plot(ZTCFQ.Time,DELTAQ.LHonClubAngularPower);
plot(ZTCFQ.Time,DELTAQ.RHonClubAngularPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Power','RS Angular Power','LE Angular Power','RE Angular Power','LH Angular Power','RH Angular Power');
legend('Location','southeast');
%Add a Title
title('Angular Power on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Angular Power');
pause(PauseTime);
%Close Figure
close(502);

%% ====== End of SCRIPT_502_3D_PLOT_DELTA_AngularPower.m ======

%% ====== Start of SCRIPT_503_3D_PLOT_DELTA_LinearPower.m ======
figure(503);
hold on;
plot(ZTCFQ.Time,DELTAQ.LSonArmLinearPower);
plot(ZTCFQ.Time,DELTAQ.RSonArmLinearPower);
plot(ZTCFQ.Time,DELTAQ.LEonForearmLinearPower);
plot(ZTCFQ.Time,DELTAQ.REonForearmLinearPower);
plot(ZTCFQ.Time,DELTAQ.LHonClubLinearPower);
plot(ZTCFQ.Time,DELTAQ.RHonClubLinearPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Power','RS Linear Power','LE Linear Power','RE Linear Power','LH Linear Power','RH Linear Power');
legend('Location','southeast');
%Add a Title
title('Linear Power on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Power');
pause(PauseTime);
%Close Figure
close(503);

%% ====== End of SCRIPT_503_3D_PLOT_DELTA_LinearPower.m ======

%% ====== Start of SCRIPT_504_3D_PLOT_DELTA_LinearWork.m ======
figure(504);
hold on;
plot(ZTCFQ.Time,DELTAQ.LSLinearWorkonArm);
plot(ZTCFQ.Time,DELTAQ.RSLinearWorkonArm);
plot(ZTCFQ.Time,DELTAQ.LELinearWorkonForearm);
plot(ZTCFQ.Time,DELTAQ.RELinearWorkonForearm);
plot(ZTCFQ.Time,DELTAQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,DELTAQ.RHLinearWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Work','RS Linear Work','LE Linear Work','RE Linear Work','LH Linear Work','LH Linear Work');
legend('Location','southeast');
%Add a Title
title('Linear Work on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Work on Distal');
pause(PauseTime);
%Close Figure
close(504);

%% ====== End of SCRIPT_504_3D_PLOT_DELTA_LinearWork.m ======

%% ====== Start of SCRIPT_505_3D_PLOT_DELTA_JointTorqueInputs.m ======
figure(505);
hold on;
plot(DELTAQ.Time,DELTAQ.HipTorqueXInput);
plot(DELTAQ.Time,DELTAQ.HipTorqueYInput);
plot(DELTAQ.Time,DELTAQ.HipTorqueZInput);
plot(DELTAQ.Time,DELTAQ.TranslationForceXInput);
plot(DELTAQ.Time,DELTAQ.TranslationForceYInput);
plot(DELTAQ.Time,DELTAQ.TranslationForceZInput);
plot(DELTAQ.Time,DELTAQ.TorsoTorqueInput);
plot(DELTAQ.Time,DELTAQ.SpineTorqueXInput);
plot(DELTAQ.Time,DELTAQ.SpineTorqueYInput);
plot(DELTAQ.Time,DELTAQ.LScapTorqueXInput);
plot(DELTAQ.Time,DELTAQ.LScapTorqueYInput);
plot(DELTAQ.Time,DELTAQ.RScapTorqueXInput);
plot(DELTAQ.Time,DELTAQ.RScapTorqueYInput);
plot(DELTAQ.Time,DELTAQ.LSTorqueXInput);
plot(DELTAQ.Time,DELTAQ.LSTorqueYInput);
plot(DELTAQ.Time,DELTAQ.LSTorqueZInput);
plot(DELTAQ.Time,DELTAQ.RSTorqueXInput);
plot(DELTAQ.Time,DELTAQ.RSTorqueYInput);
plot(DELTAQ.Time,DELTAQ.RSTorqueZInput);
plot(DELTAQ.Time,DELTAQ.LETorqueInput);
plot(DELTAQ.Time,DELTAQ.RETorqueInput);
plot(DELTAQ.Time,DELTAQ.LFTorqueInput);
plot(DELTAQ.Time,DELTAQ.RFTorqueInput);
plot(DELTAQ.Time,DELTAQ.LWTorqueXInput);
plot(DELTAQ.Time,DELTAQ.LWTorqueYInput);
plot(DELTAQ.Time,DELTAQ.RWTorqueXInput);
plot(DELTAQ.Time,DELTAQ.RWTorqueYInput);
ylabel('Torque (Nm)');
grid 'on';
%Add Legend to Plot
legend('Hip Torque X','Hip Torque Y','Hip Torque Z','Translation Force X',...
'Translation Force Y','Translation Force Z','Torso Torque','Spine Torque X',...
'Spine Torque Y','LScap Torque X','Left Scap Torque Y','RScap Torque X',...
'RScapTorqueY','LS Torque X','LS Torque Y','LS Torque Z','RS Torque X','RS Torque Y',...
'RS Torque Z','LE Torque','RE Torque','LF Torque','RF Torque','LW Torque X',...
'LW Torque Y','RW Torque X','RW Torque Y');
legend('Location','southeast');
%Add a Title
title('Joint Torque Inputs');
%subtitle('Left Hand, Right Hand, Total');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Joint Torque Inputs');
pause(PauseTime);
%Close Figure
close(505);

%% ====== End of SCRIPT_505_3D_PLOT_DELTA_JointTorqueInputs.m ======

%% ====== Start of SCRIPT_506_3D_PLOT_DELTA_TotalWork.m ======
figure(506);
hold on;
plot(DELTAQ.Time,DELTAQ.TotalLSWork);
plot(DELTAQ.Time,DELTAQ.TotalRSWork);
plot(DELTAQ.Time,DELTAQ.TotalLEWork);
plot(DELTAQ.Time,DELTAQ.TotalREWork);
plot(DELTAQ.Time,DELTAQ.TotalLHWork);
plot(DELTAQ.Time,DELTAQ.TotalRHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Total Work','RS Total Work','LE Total Work','RE Total Work','LH Total Work','RH Total Work');
legend('Location','southeast');
%Add a Title
title('Total Work on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Total Work');
pause(PauseTime);
%Close Figure
close(506);

%% ====== End of SCRIPT_506_3D_PLOT_DELTA_TotalWork.m ======

%% ====== Start of SCRIPT_507_3D_PLOT_DELTA_TotalPower.m ======
figure(507);
hold on;
plot(DELTAQ.Time,DELTAQ.TotalLSPower);
plot(DELTAQ.Time,DELTAQ.TotalRSPower);
plot(DELTAQ.Time,DELTAQ.TotalLEPower);
plot(DELTAQ.Time,DELTAQ.TotalREPower);
plot(DELTAQ.Time,DELTAQ.TotalLHPower);
plot(DELTAQ.Time,DELTAQ.TotalRHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Total Power','RS Total Power','LE Total Power','RE Total Power','LH Total Power','RH Total Power');
legend('Location','southeast');
%Add a Title
title('Total Power on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Total Power');
pause(PauseTime);
%Close Figure
close(507);

%% ====== End of SCRIPT_507_3D_PLOT_DELTA_TotalPower.m ======

%% ====== Start of SCRIPT_522_3D_PLOT_DELTA_ForceAlongHandPath.m ======
figure(522);
plot(ZTCFQ.Time,DELTAQ.ForceAlongHandPath);
xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('Force Along Hand Path');
legend('Location','southeast');
%Add a Title
title('Force Along Hand Path');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Force Along Hand Path');
pause(PauseTime);
%Close Figure
close(522);

%% ====== End of SCRIPT_522_3D_PLOT_DELTA_ForceAlongHandPath.m ======

%% ====== Start of SCRIPT_525_3D_PLOT_DELTA_LHRHForceAlongHandPath.m ======
figure(525);
hold on;
plot(ZTCFQ.Time,DELTAQ.LHForceAlongHandPath);
plot(ZTCFQ.Time,DELTAQ.RHForceAlongHandPath);
plot(ZTCFQ.Time,DELTAQ.ForceAlongHandPath);
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('LH Force on Left Hand Path','RH Force on Right Hand Path','Net Force Along MP Hand Path');
legend('Location','southeast');
%Add a Title
title('Force Along Hand Path');
subtitle('DELTA');
%subtitle('Left Hand, Right Hand, Total');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Force Along Hand Path - LH RH Total');
pause(PauseTime);
%Close Figure
close(525);

%% ====== End of SCRIPT_525_3D_PLOT_DELTA_LHRHForceAlongHandPath.m ======

%% ====== Start of SCRIPT_526_3D_PLOT_DELTA_LinearImpulse.m ======
figure(526);
plot(ZTCFQ.Time,DELTAQ.LinearImpulseonClub);
xlabel('Time (s)');
ylabel('Impulse (Ns)');
grid 'on';
%Add Legend to Plot
legend('Linear Impulse');
legend('Location','southeast');
%Add a Title
title('Linear Impulse');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Impulse');
pause(PauseTime);
%Close Figure
close(526);

%% ====== End of SCRIPT_526_3D_PLOT_DELTA_LinearImpulse.m ======

%% ====== Start of SCRIPT_527_3D_PLOT_DELTA_LinearWork.m ======
figure(527);
hold on;
plot(ZTCFQ.Time,DELTAQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,DELTAQ.RHLinearWorkonClub);
plot(ZTCFQ.Time,DELTAQ.LinearWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Work','RH Linear Work','Net Force Linear Work (midpoint)');
legend('Location','southeast');
%Add a Title
title('Linear Work');
subtitle('DELTA');
%subtitle('Left Hand, Right Hand, Total');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Work');
pause(PauseTime);
%Close Figure
close(527);

%% ====== End of SCRIPT_527_3D_PLOT_DELTA_LinearWork.m ======

%% ====== Start of SCRIPT_528_3D_PLOT_DELTA_LinearImpulse.m ======
figure(528);
hold on;
plot(ZTCFQ.Time,DELTAQ.LHLinearImpulseonClub);
plot(ZTCFQ.Time,DELTAQ.RHLinearImpulseonClub);
plot(ZTCFQ.Time,DELTAQ.LinearImpulseonClub);
ylabel('Linear Impulse (kgm/s)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Impulse','RH Linear Impulse','Net Force Linear Impulse (midpoint)');
legend('Location','southeast');
%Add a Title
title('Linear Impulse');
subtitle('DELTA');
%subtitle('Left Hand, Right Hand, Total');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Impulse LH,RH,Total');
pause(PauseTime);
%Close Figure
close(528);

%% ====== End of SCRIPT_528_3D_PLOT_DELTA_LinearImpulse.m ======

%% ====== Start of SCRIPT_529_3D_PLOT_DELTA_EquivalentCoupleandMOF.m ======
figure(529);
hold on;
EQMPLOCAL=DELTAQ.EquivalentMidpointCoupleLocal(:,3);
MPMOFLOCAL=DELTAQ.MPMOFonClubLocal(:,3);
SUMOFMOMENTSLOCAL=DELTAQ.SumofMomentsonClubLocal(:,3);
plot(ZTCFQ.Time,EQMPLOCAL);
plot(ZTCFQ.Time,MPMOFLOCAL);
plot(ZTCFQ.Time,SUMOFMOMENTSLOCAL);
ylabel('Torque (Nm)');
grid 'on';
%Add Legend to Plot
legend('Equivalent Midpoint Couple','Total Force on Midpoint MOF','Sum of Moments');
legend('Location','southeast');
%Add a Title
title('Equivalent Couple, Moment of Force, Sum of Moments');
subtitle('DELTA - Grip Reference Frame');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Equivalent Couple and MOF');
pause(PauseTime);
%Close Figure
close(529);

%% ====== End of SCRIPT_529_3D_PLOT_DELTA_EquivalentCoupleandMOF.m ======

%% ====== Start of SCRIPT_530_3D_PLOT_DELTA_LSWork.m ======
figure(530);
hold on;
plot(DELTAQ.Time,DELTAQ.LSLinearWorkonArm);
plot(DELTAQ.Time,DELTAQ.LSAngularWorkonArm);
plot(DELTAQ.Time,DELTAQ.TotalLSWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Work','LS Angular Work','LS Total Work');
legend('Location','southeast');
%Add a Title
title('Left Shoulder Work on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Shoulder Work');
pause(PauseTime);
%Close Figure
close(530);

%% ====== End of SCRIPT_530_3D_PLOT_DELTA_LSWork.m ======

%% ====== Start of SCRIPT_531_3D_PLOT_DELTA_LSPower.m ======
figure(531);
hold on;
plot(DELTAQ.Time,DELTAQ.LSonArmLinearPower);
plot(DELTAQ.Time,DELTAQ.LSonArmAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalLSPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Power','LS Angular Power','LS Total Power');
legend('Location','southeast');
%Add a Title
title('Left Shoulder Power on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Shoulder Power');
pause(PauseTime);
%Close Figure
close(531);

%% ====== End of SCRIPT_531_3D_PLOT_DELTA_LSPower.m ======

%% ====== Start of SCRIPT_532_3D_PLOT_DELTA_RSWork.m ======
figure(532);
hold on;
plot(DELTAQ.Time,DELTAQ.RSLinearWorkonArm);
plot(DELTAQ.Time,DELTAQ.RSAngularWorkonArm);
plot(DELTAQ.Time,DELTAQ.TotalRSWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RS Linear Work','RS Angular Work','RS Total Work');
legend('Location','southeast');
%Add a Title
title('Right Shoulder Work on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Shoulder Work');
pause(PauseTime);
%Close Figure
close(532);

%% ====== End of SCRIPT_532_3D_PLOT_DELTA_RSWork.m ======

%% ====== Start of SCRIPT_533_3D_PLOT_DELTA_RSPower.m ======
figure(533);
hold on;
plot(DELTAQ.Time,DELTAQ.RSonArmLinearPower);
plot(DELTAQ.Time,DELTAQ.RSonArmAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalRSPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RS Linear Power','RS Angular Power','RS Total Power');
legend('Location','southeast');
%Add a Title
title('Right Shoulder Power on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Shoulder Power');
pause(PauseTime);
%Close Figure
close(533);

%% ====== End of SCRIPT_533_3D_PLOT_DELTA_RSPower.m ======

%% ====== Start of SCRIPT_534_3D_PLOT_DELTA_LEWork.m ======
figure(534);
hold on;
plot(DELTAQ.Time,DELTAQ.LELinearWorkonForearm);
plot(DELTAQ.Time,DELTAQ.LEAngularWorkonForearm);
plot(DELTAQ.Time,DELTAQ.TotalLEWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LE Linear Work','LE Angular Work','LE Total Work');
legend('Location','southeast');
%Add a Title
title('Left Elbow Work on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Elbow Work');
pause(PauseTime);
%Close Figure
close(534);

%% ====== End of SCRIPT_534_3D_PLOT_DELTA_LEWork.m ======

%% ====== Start of SCRIPT_535_3D_PLOT_DELTA_LEPower.m ======
figure(535);
hold on;
plot(DELTAQ.Time,DELTAQ.LEonForearmLinearPower);
plot(DELTAQ.Time,DELTAQ.LEonForearmAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalLEPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LE Linear Power','LE Angular Power','LE Total Power');
legend('Location','southeast');
%Add a Title
title('Left Elbow Power on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Elbow Power');
pause(PauseTime);
%Close Figure
close(535);

%% ====== End of SCRIPT_535_3D_PLOT_DELTA_LEPower.m ======

%% ====== Start of SCRIPT_536_3D_PLOT_DELTA_REWork.m ======
figure(536);
hold on;
plot(DELTAQ.Time,DELTAQ.RELinearWorkonForearm);
plot(DELTAQ.Time,DELTAQ.REAngularWorkonForearm);
plot(DELTAQ.Time,DELTAQ.TotalREWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RE Linear Work','RE Angular Work','RE Total Work');
legend('Location','southeast');
%Add a Title
title('Right Elbow Work on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Elbow Work');
pause(PauseTime);
%Close Figure
close(536);

%% ====== End of SCRIPT_536_3D_PLOT_DELTA_REWork.m ======

%% ====== Start of SCRIPT_537_3D_PLOT_DELTA_REPower.m ======
figure(537);
hold on;
plot(DELTAQ.Time,DELTAQ.REonForearmLinearPower);
plot(DELTAQ.Time,DELTAQ.REonForearmAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalREPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RE Linear Power','RE Angular Power','RE Total Power');
legend('Location','southeast');
%Add a Title
title('Right Elbow Power on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Elbow Power');
pause(PauseTime);
%Close Figure
close(537);

%% ====== End of SCRIPT_537_3D_PLOT_DELTA_REPower.m ======

%% ====== Start of SCRIPT_538_3D_PLOT_DELTA_LHWork.m ======
figure(538);
hold on;
plot(DELTAQ.Time,DELTAQ.LHLinearWorkonClub);
plot(DELTAQ.Time,DELTAQ.LHAngularWorkonClub);
plot(DELTAQ.Time,DELTAQ.TotalLHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Work','LH Angular Work','LH Total Work');
legend('Location','southeast');
%Add a Title
title('Left Wrist Work on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Wrist Work');
pause(PauseTime);
%Close Figure
close(538);

%% ====== End of SCRIPT_538_3D_PLOT_DELTA_LHWork.m ======

%% ====== Start of SCRIPT_539_3D_PLOT_DELTA_LHPower.m ======
figure(539);
hold on;
plot(DELTAQ.Time,DELTAQ.LHonClubLinearPower);
plot(DELTAQ.Time,DELTAQ.LHonClubAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalLHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Power','LH Angular Power','LH Total Power');
legend('Location','southeast');
%Add a Title
title('Left Wrist Power on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Wrist Power');
pause(PauseTime);
%Close Figure
close(539);

%% ====== End of SCRIPT_539_3D_PLOT_DELTA_LHPower.m ======

%% ====== Start of SCRIPT_540_3D_PLOT_DELTA_RHWork.m ======
figure(540);
hold on;
plot(DELTAQ.Time,DELTAQ.RHLinearWorkonClub);
plot(DELTAQ.Time,DELTAQ.RHAngularWorkonClub);
plot(DELTAQ.Time,DELTAQ.TotalRHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RH Linear Work','RH Angular Work','RH Total Work');
legend('Location','southeast');
%Add a Title
title('Right Wrist Work on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Wrist Work');
pause(PauseTime);
%Close Figure
close(540);

%% ====== End of SCRIPT_540_3D_PLOT_DELTA_RHWork.m ======

%% ====== Start of SCRIPT_541_3D_PLOT_DELTA_RHPower.m ======
figure(541);
hold on;
plot(DELTAQ.Time,DELTAQ.RHonClubLinearPower);
plot(DELTAQ.Time,DELTAQ.RHonClubAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalRHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RH Linear Power','RH Angular Power','RH Total Power');
legend('Location','southeast');
%Add a Title
title('Right Wrist Power on Distal Segment');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Wrist Power');
pause(PauseTime);
%Close Figure
close(541);

%% ====== End of SCRIPT_541_3D_PLOT_DELTA_RHPower.m ======

%% ====== Start of SCRIPT_548_3D_PLOT_DELTA_LocalHandForces.m ======
figure(548);
hold on;
plot(DELTAQ.Time(:,1),DELTAQ.LHonClubForceLocal(:,1));
plot(DELTAQ.Time(:,1),DELTAQ.LHonClubForceLocal(:,2));
plot(DELTAQ.Time(:,1),DELTAQ.LHonClubForceLocal(:,3));
plot(DELTAQ.Time(:,1),DELTAQ.RHonClubForceLocal(:,1));
plot(DELTAQ.Time(:,1),DELTAQ.RHonClubForceLocal(:,2));
plot(DELTAQ.Time(:,1),DELTAQ.RHonClubForceLocal(:,3));
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('Left Hand X','Left Hand Y','Left Hand Z','Right Hand X','Right Hand Y','Right Hand Z');
legend('Location','southeast');
%Add a Title
title('Local Hand Forces on Club');
subtitle('DELTA');
%Save Figure
savefig('Delta Charts/DELTA_Plot - Local Hand Forces');
pause(PauseTime);
%Close Figure
close(548);

%% ====== End of SCRIPT_548_3D_PLOT_DELTA_LocalHandForces.m ======

%% ====== Start of SCRIPT_550_3D_PLOT_DELTA_TorqueInputvsTorqueOutput.m ======
figure(550);
hold on;
plot(DELTAQ.Time,DELTAQ.HipConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.HipConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.HipConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.TorsoConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.TorsoConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.TorsoConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.SpineConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.SpineConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.SpineConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.LScapConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.LScapConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.LScapConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.RScapConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.RScapConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.RScapConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.LSConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.LSConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.LSConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.RSConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.RSConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.RSConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.LEConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.LEConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.LEConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.REConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.REConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.REConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.LFConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.LFConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.LFConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.RFConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.RFConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.RFConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.LWConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.LWConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.LWConstraintTorqueZ);
plot(DELTAQ.Time,DELTAQ.RWConstraintTorqueX);
plot(DELTAQ.Time,DELTAQ.RWConstraintTorqueY);
plot(DELTAQ.Time,DELTAQ.RWConstraintTorqueZ);
ylabel('Torque (Nm)');
grid 'on';
%Add Legend to Plot
legend('Hip Constraint Torque X','Hip Constraint Torque Y','Hip Constraint Torque Z',...
'Torso Constraint Torque X','Torso Constraint Torque Y','Torso Constraint Torque Z',...
'Spine Constraint Torque X','Spine Constraint Torque Y','Spine Constraint Torque Z',...
'LScap Constraint Torque X','LScap Constraint Torque Y','LScap Constraint Torque Z',...
'RScap Constraint Torque X','RScap Constraint Torque Y','RScap Constraint Torque Z',...
'LS Constraint Torque X','LS Constraint Torque Y','LS Constraint Torque Z',...
'RS Constraint Torque X','RS Constraint Torque Y','RS Constraint Torque Z',...
'LE Constraint Torque X','LE Constraint Torque Y','LE Constraint Torque Z',...
'RE Constraint Torque X','RE Constraint Torque Y','RE Constraint Torque Z',...
'LF Constraint Torque X','LF Constraint Torque Y','LF Constraint Torque Z',...
'RF Constraint Torque X','RF Constraint Torque Y','RF Constraint Torque Z',...
'LW Constraint Torque X','LW Constraint Torque Y','LW Constraint Torque Z',...
'RW Constraint Torque X','RW Constraint Torque Y','RW Constraint Torque Z');
legend('Location','southeast');
%Add a Title
title('Joint Torque Inputs');
subtitle('DELTA');
%Save Figure
savefig('DELTA Charts/DELTA_Plot - Joint Torque Inputs');
pause(PauseTime);
%Close Figure
close(550);

%% ====== End of SCRIPT_550_3D_PLOT_DELTA_TorqueInputvsTorqueOutput.m ======

end
