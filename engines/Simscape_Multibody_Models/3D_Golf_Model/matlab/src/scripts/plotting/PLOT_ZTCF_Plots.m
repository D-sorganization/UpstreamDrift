function PLOT_ZTCF_Plots(ZTCFQ)
%%% ====== Start of SCRIPT_301_3D_PLOT_ZTCF_AngularWork.m ======
figure(301);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LSAngularWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.RSAngularWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.LEAngularWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.REAngularWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.LHAngularWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.RHAngularWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Work','RS Angular Work','LE Angular Work','RE Angular Work','LH Angular Work','RH Angular Work');
legend('Location','southeast');
%Add a Title
title('Angular Work on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Angular Work');
pause(PauseTime);
%Close Figure
close(301);

%% ====== End of SCRIPT_301_3D_PLOT_ZTCF_AngularWork.m ======

%% ====== Start of SCRIPT_302_3D_PLOT_ZTCF_AngularPower.m ======
figure(302);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LSonArmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.RSonArmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.LEonForearmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.REonForearmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.LHonClubAngularPower);
plot(ZTCFQ.Time,ZTCFQ.RHonClubAngularPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Power','RS Angular Power','LE Angular Power','RE Angular Power','LH Angular Power','RH Angular Power');
legend('Location','southeast');
%Add a Title
title('Angular Power on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Angular Power');
pause(PauseTime);
%Close Figure
close(302);

%% ====== End of SCRIPT_302_3D_PLOT_ZTCF_AngularPower.m ======

%% ====== Start of SCRIPT_303_3D_PLOT_ZTCF_LinearPower.m ======
figure(303);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LSonArmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.RSonArmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LEonForearmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.REonForearmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LHonClubLinearPower);
plot(ZTCFQ.Time,ZTCFQ.RHonClubLinearPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Power','RS Linear Power','LE Linear Power','RE Linear Power','LH Linear Power','RH Linear Power');
legend('Location','southeast');
%Add a Title
title('Linear Power on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Linear Power');
pause(PauseTime);
%Close Figure
close(303);

%% ====== End of SCRIPT_303_3D_PLOT_ZTCF_LinearPower.m ======

%% ====== Start of SCRIPT_304_3D_PLOT_ZTCF_LinearWork.m ======
figure(304);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LSLinearWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.RSLinearWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.LELinearWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.RELinearWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Work','RS Linear Work','LE Linear Work','RE Linear Work','LW Linear Work','RH Linear Work');
legend('Location','southeast');
%Add a Title
title('Linear Work on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Linear Work on Distal');
pause(PauseTime);
%Close Figure
close(304);

%% ====== End of SCRIPT_304_3D_PLOT_ZTCF_LinearWork.m ======

%% ====== Start of SCRIPT_305_3D_PLOT_ZTCF_JointTorqueInputs.m ======
figure(305);
hold on;
plot(ZTCFQ.Time,ZTCFQ.HipTorqueXInput);
plot(ZTCFQ.Time,ZTCFQ.HipTorqueYInput);
plot(ZTCFQ.Time,ZTCFQ.HipTorqueZInput);
plot(ZTCFQ.Time,ZTCFQ.TranslationForceXInput);
plot(ZTCFQ.Time,ZTCFQ.TranslationForceYInput);
plot(ZTCFQ.Time,ZTCFQ.TranslationForceZInput);
plot(ZTCFQ.Time,ZTCFQ.TorsoTorqueInput);
plot(ZTCFQ.Time,ZTCFQ.SpineTorqueXInput);
plot(ZTCFQ.Time,ZTCFQ.SpineTorqueYInput);
plot(ZTCFQ.Time,ZTCFQ.LScapTorqueXInput);
plot(ZTCFQ.Time,ZTCFQ.LScapTorqueYInput);
plot(ZTCFQ.Time,ZTCFQ.RScapTorqueXInput);
plot(ZTCFQ.Time,ZTCFQ.RScapTorqueYInput);
plot(ZTCFQ.Time,ZTCFQ.LSTorqueXInput);
plot(ZTCFQ.Time,ZTCFQ.LSTorqueYInput);
plot(ZTCFQ.Time,ZTCFQ.LSTorqueZInput);
plot(ZTCFQ.Time,ZTCFQ.RSTorqueXInput);
plot(ZTCFQ.Time,ZTCFQ.RSTorqueYInput);
plot(ZTCFQ.Time,ZTCFQ.RSTorqueZInput);
plot(ZTCFQ.Time,ZTCFQ.LETorqueInput);
plot(ZTCFQ.Time,ZTCFQ.RETorqueInput);
plot(ZTCFQ.Time,ZTCFQ.LFTorqueInput);
plot(ZTCFQ.Time,ZTCFQ.RFTorqueInput);
plot(ZTCFQ.Time,ZTCFQ.LWTorqueXInput);
plot(ZTCFQ.Time,ZTCFQ.LWTorqueYInput);
plot(ZTCFQ.Time,ZTCFQ.RWTorqueXInput);
plot(ZTCFQ.Time,ZTCFQ.RWTorqueYInput);
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
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Joint Torque Inputs');
pause(PauseTime);
%Close Figure
close(305);

%% ====== End of SCRIPT_305_3D_PLOT_ZTCF_JointTorqueInputs.m ======

%% ====== Start of SCRIPT_306_3D_PLOT_ZTCF_TotalWork.m ======
figure(306);
hold on;
plot(ZTCFQ.Time,ZTCFQ.TotalLSWork);
plot(ZTCFQ.Time,ZTCFQ.TotalRSWork);
plot(ZTCFQ.Time,ZTCFQ.TotalLEWork);
plot(ZTCFQ.Time,ZTCFQ.TotalREWork);
plot(ZTCFQ.Time,ZTCFQ.TotalLHWork);
plot(ZTCFQ.Time,ZTCFQ.TotalRHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Total Work','RS Total Work','LE Total Work','RE Total Work','LH Total Work','RH Total Work');
legend('Location','southeast');
%Add a Title
title('Total Work on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Total Work');
pause(PauseTime);
%Close Figure
close(306);

%% ====== End of SCRIPT_306_3D_PLOT_ZTCF_TotalWork.m ======

%% ====== Start of SCRIPT_307_3D_PLOT_ZTCF_TotalPower.m ======
figure(307);
hold on;
plot(ZTCFQ.Time,ZTCFQ.TotalLSPower);
plot(ZTCFQ.Time,ZTCFQ.TotalRSPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLEPower);
plot(ZTCFQ.Time,ZTCFQ.TotalREPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLHPower);
plot(ZTCFQ.Time,ZTCFQ.TotalRHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Total Power','RS Total Power','LE Total Power','RE Total Power','LH Total Power','RH Total Power');
legend('Location','southeast');
%Add a Title
title('Total Power on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Total Power');
pause(PauseTime);
%Close Figure
close(307);

%% ====== End of SCRIPT_307_3D_PLOT_ZTCF_TotalPower.m ======

%% ====== Start of SCRIPT_322_3D_PLOT_ZTCF_ForceAlongHandPath.m ======
figure(322);
plot(ZTCFQ.Time,ZTCFQ.ForceAlongHandPath);
xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('Force Along Hand Path');
legend('Location','southeast');
%Add a Title
title('Force Along Hand Path');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Force Along Hand Path');
pause(PauseTime);
%Close Figure
close(322);

%% ====== End of SCRIPT_322_3D_PLOT_ZTCF_ForceAlongHandPath.m ======

%% ====== Start of SCRIPT_323_3D_PLOT_ZTCF_CHSandHandSpeed.m ======
figure(323);
hold on;
plot(ZTCFQ.Time,ZTCFQ.("CHS (mph)"));
plot(ZTCFQ.Time,ZTCFQ.("Hand Speed (mph)"));
xlabel('Time (s)');
ylabel('Speed (mph)');
grid 'on';
%Add Legend to Plot
legend('Clubhead Speed (mph)','Hand Speed (mph)');
legend('Location','southeast');
%Add a Title
title('Clubhead and Hand Speed');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - CHS and Hand Speed');
pause(PauseTime);
%Close Figure
close(323);

%% ====== End of SCRIPT_323_3D_PLOT_ZTCF_CHSandHandSpeed.m ======

%% ====== Start of SCRIPT_325_3D_PLOT_ZTCF_LHRHForceAlongHandPath.m ======
figure(325);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LHForceAlongHandPath);
plot(ZTCFQ.Time,ZTCFQ.RHForceAlongHandPath);
plot(ZTCFQ.Time,ZTCFQ.ForceAlongHandPath);
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('LH Force on Left Hand Path','RH Force on Right Hand Path','Net Force Along MP Hand Path');
legend('Location','southeast');
%Add a Title
title('Force Along Hand Path');
subtitle('ZTCF');
%subtitle('Left Hand, Right Hand, Total');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Force Along Hand Path - LH RH Total');
pause(PauseTime);
%Close Figure
close(325);

%% ====== End of SCRIPT_325_3D_PLOT_ZTCF_LHRHForceAlongHandPath.m ======

%% ====== Start of SCRIPT_326_3D_PLOT_ZTCF_LinearImpulse.m ======
figure(326);
plot(ZTCFQ.Time,ZTCFQ.LinearImpulseonClub);
xlabel('Time (s)');
ylabel('Impulse (Ns)');
grid 'on';
%Add Legend to Plot
legend('Linear Impulse');
legend('Location','southeast');
%Add a Title
title('Linear Impulse');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Linear Impulse');
pause(PauseTime);
%Close Figure
close(326);

%% ====== End of SCRIPT_326_3D_PLOT_ZTCF_LinearImpulse.m ======

%% ====== Start of SCRIPT_327_3D_PLOT_ZTCF_LinearWork.m ======
figure(327);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.LinearWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Work','RH Linear Work','Net Force Linear Work (midpoint)');
legend('Location','southeast');
%Add a Title
title('Linear Work');
subtitle('ZTCF');
%subtitle('Left Hand, Right Hand, Total');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Linear Work');
pause(PauseTime);
%Close Figure
close(327);

%% ====== End of SCRIPT_327_3D_PLOT_ZTCF_LinearWork.m ======

%% ====== Start of SCRIPT_328_3D_PLOT_ZTCF_LinearImpulse.m ======
figure(328);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LHLinearImpulseonClub);
plot(ZTCFQ.Time,ZTCFQ.RHLinearImpulseonClub);
plot(ZTCFQ.Time,ZTCFQ.LinearImpulseonClub);
ylabel('Linear Impulse (kgm/s)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Impulse','RH Linear Impulse','Net Force Linear Impulse (midpoint)');
legend('Location','southeast');
%Add a Title
title('Linear Impulse');
subtitle('ZTCF');
%subtitle('Left Hand, Right Hand, Total');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Linear Impulse LH,RH,Total');
pause(PauseTime);
%Close Figure
close(328);

%% ====== End of SCRIPT_328_3D_PLOT_ZTCF_LinearImpulse.m ======

%% ====== Start of SCRIPT_329_3D_PLOT_ZTCF_EquivalentCoupleandMOF.m ======
figure(329);
hold on;
EQMPLOCAL=ZTCFQ.EquivalentMidpointCoupleLocal(:,3);
MPMOFLOCAL=ZTCFQ.MPMOFonClubLocal(:,3);
SUMOFMOMENTSLOCAL=ZTCFQ.SumofMomentsonClubLocal(:,3);
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
subtitle('ZTCF - Grip Reference Frame');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Equivalent Couple and MOF');
pause(PauseTime);
%Close Figure
close(329);

%% ====== End of SCRIPT_329_3D_PLOT_ZTCF_EquivalentCoupleandMOF.m ======

%% ====== Start of SCRIPT_330_3D_PLOT_ZTCF_LSWork.m ======
figure(330);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LSLinearWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.LSAngularWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.TotalLSWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Work','LS Angular Work','LS Total Work');
legend('Location','southeast');
%Add a Title
title('Left Shoulder Work on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Shoulder Work');
pause(PauseTime);
%Close Figure
close(330);

%% ====== End of SCRIPT_330_3D_PLOT_ZTCF_LSWork.m ======

%% ====== Start of SCRIPT_331_3D_PLOT_ZTCF_LSPower.m ======
figure(931);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LSonArmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LSonArmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLSPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Power','LS Angular Power','LS Total Power');
legend('Location','southeast');
%Add a Title
title('Left Shoulder Power on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Shoulder Power');
pause(PauseTime);
%Close Figure
close(931);

%% ====== End of SCRIPT_331_3D_PLOT_ZTCF_LSPower.m ======

%% ====== Start of SCRIPT_332_3D_PLOT_ZTCF_RSWork.m ======
figure(332);
hold on;
plot(ZTCFQ.Time,ZTCFQ.RSLinearWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.RSAngularWorkonArm);
plot(ZTCFQ.Time,ZTCFQ.TotalRSWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RS Linear Work','RS Angular Work','RS Total Work');
legend('Location','southeast');
%Add a Title
title('Right Shoulder Work on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Shoulder Work');
pause(PauseTime);
%Close Figure
close(332);

%% ====== End of SCRIPT_332_3D_PLOT_ZTCF_RSWork.m ======

%% ====== Start of SCRIPT_333_3D_PLOT_ZTCF_RSPower.m ======
figure(333);
hold on;
plot(ZTCFQ.Time,ZTCFQ.RSonArmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.RSonArmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalRSPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RS Linear Power','RS Angular Power','RS Total Power');
legend('Location','southeast');
%Add a Title
title('Right Shoulder Power on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Shoulder Power');
pause(PauseTime);
%Close Figure
close(333);

%% ====== End of SCRIPT_333_3D_PLOT_ZTCF_RSPower.m ======

%% ====== Start of SCRIPT_334_3D_PLOT_ZTCF_LEWork.m ======
figure(334);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LELinearWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.LEAngularWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.TotalLEWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LE Linear Work','LE Angular Work','LE Total Work');
legend('Location','southeast');
%Add a Title
title('Left Elbow Work on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Elbow Work');
pause(PauseTime);
%Close Figure
close(334);

%% ====== End of SCRIPT_334_3D_PLOT_ZTCF_LEWork.m ======

%% ====== Start of SCRIPT_335_3D_PLOT_ZTCF_LEPower.m ======
figure(335);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LEonForearmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LEonForearmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLEPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LE Linear Power','LE Angular Power','LE Total Power');
legend('Location','southeast');
%Add a Title
title('Left Elbow Power on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Elbow Power');
pause(PauseTime);
%Close Figure
close(335);

%% ====== End of SCRIPT_335_3D_PLOT_ZTCF_LEPower.m ======

%% ====== Start of SCRIPT_336_3D_PLOT_ZTCF_REWork.m ======
figure(336);
hold on;
plot(ZTCFQ.Time,ZTCFQ.RELinearWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.REAngularWorkonForearm);
plot(ZTCFQ.Time,ZTCFQ.TotalREWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RE Linear Work','RE Angular Work','RE Total Work');
legend('Location','southeast');
%Add a Title
title('Right Elbow Work on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Elbow Work');
pause(PauseTime);
%Close Figure
close(336);

%% ====== End of SCRIPT_336_3D_PLOT_ZTCF_REWork.m ======

%% ====== Start of SCRIPT_337_3D_PLOT_ZTCF_REPower.m ======
figure(337);
hold on;
plot(ZTCFQ.Time,ZTCFQ.REonForearmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.REonForearmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalREPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RE Linear Power','RE Angular Power','RE Total Power');
legend('Location','southeast');
%Add a Title
title('Right Elbow Power on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Elbow Power');
pause(PauseTime);
%Close Figure
close(337);

%% ====== End of SCRIPT_337_3D_PLOT_ZTCF_REPower.m ======

%% ====== Start of SCRIPT_338_3D_PLOT_ZTCF_LHWork.m ======
figure(338);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.LHAngularWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.TotalLHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Work','LH Angular Work','LH Total Work');
legend('Location','southeast');
%Add a Title
title('Left Wrist Work on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Wrist Work');
pause(PauseTime);
%Close Figure
close(338);

%% ====== End of SCRIPT_338_3D_PLOT_ZTCF_LHWork.m ======

%% ====== Start of SCRIPT_339_3D_PLOT_ZTCF_LHPower.m ======
figure(339);
hold on;
plot(ZTCFQ.Time,ZTCFQ.LHonClubLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LHonClubAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Power','LH Angular Power','LH Total Power');
legend('Location','southeast');
%Add a Title
title('Left Wrist Power on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Wrist Power');
pause(PauseTime);
%Close Figure
close(339);

%% ====== End of SCRIPT_339_3D_PLOT_ZTCF_LHPower.m ======

%% ====== Start of SCRIPT_340_3D_PLOT_ZTCF_RHWork.m ======
figure(340);
hold on;
plot(ZTCFQ.Time,ZTCFQ.RHLinearWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.RHAngularWorkonClub);
plot(ZTCFQ.Time,ZTCFQ.TotalRHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RH Linear Work','RH Angular Work','RH Total Work');
legend('Location','southeast');
%Add a Title
title('Right Wrist Work on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Wrist Work');
pause(PauseTime);
%Close Figure
close(340);

%% ====== End of SCRIPT_340_3D_PLOT_ZTCF_RHWork.m ======

%% ====== Start of SCRIPT_341_3D_PLOT_ZTCF_RHPower.m ======
figure(341);
hold on;
plot(ZTCFQ.Time,ZTCFQ.RHonClubLinearPower);
plot(ZTCFQ.Time,ZTCFQ.RHonClubAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalRHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RH Linear Power','RH Angular Power','RH Total Power');
legend('Location','southeast');
%Add a Title
title('Right Wrist Power on Distal Segment');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Wrist Power');
pause(PauseTime);
%Close Figure
close(341);

%% ====== End of SCRIPT_341_3D_PLOT_ZTCF_RHPower.m ======

%% ====== Start of SCRIPT_348_3D_PLOT_ZTCF_LocalHandForces.m ======
figure(348);
hold on;
plot(ZTCFQ.Time(:,1),ZTCFQ.LHonClubForceLocal(:,1));
plot(ZTCFQ.Time(:,1),ZTCFQ.LHonClubForceLocal(:,2));
plot(ZTCFQ.Time(:,1),ZTCFQ.LHonClubForceLocal(:,3));
plot(ZTCFQ.Time(:,1),ZTCFQ.RHonClubForceLocal(:,1));
plot(ZTCFQ.Time(:,1),ZTCFQ.RHonClubForceLocal(:,2));
plot(ZTCFQ.Time(:,1),ZTCFQ.RHonClubForceLocal(:,3));
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('Left Hand X','Left Hand Y','Left Hand Z','Right Hand X','Right Hand Y','Right Hand Z');
legend('Location','southeast');
%Add a Title
title('Local Hand Forces on Club');
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Local Hand Forces');
pause(PauseTime);
%Close Figure
close(348);

%% ====== End of SCRIPT_348_3D_PLOT_ZTCF_LocalHandForces.m ======

%% ====== Start of SCRIPT_350_3D_PLOT_ZTCF_TorqueInputvsTorqueOutput.m ======
figure(350);
hold on;
plot(ZTCFQ.Time,ZTCFQ.HipConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.HipConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.HipConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.TorsoConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.TorsoConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.TorsoConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.SpineConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.SpineConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.SpineConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.LScapConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.LScapConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.LScapConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.RScapConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.RScapConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.RScapConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.LSConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.LSConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.LSConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.RSConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.RSConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.RSConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.LEConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.LEConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.LEConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.REConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.REConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.REConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.LFConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.LFConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.LFConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.RFConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.RFConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.RFConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.LWConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.LWConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.LWConstraintTorqueZ);
plot(ZTCFQ.Time,ZTCFQ.RWConstraintTorqueX);
plot(ZTCFQ.Time,ZTCFQ.RWConstraintTorqueY);
plot(ZTCFQ.Time,ZTCFQ.RWConstraintTorqueZ);
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
subtitle('ZTCF');
%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Joint Torque Inputs');
pause(PauseTime);
%Close Figure
close(350);

%% ====== End of SCRIPT_350_3D_PLOT_ZTCF_TorqueInputvsTorqueOutput.m ======

end
