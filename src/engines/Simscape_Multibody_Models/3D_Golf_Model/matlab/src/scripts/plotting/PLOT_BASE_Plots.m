function PLOT_BASE_Plots(BASEQ)
%%% ====== Start of SCRIPT_101_3D_PLOT_BaseData_AngularWork.m ======
figure(101);
hold on;
plot(BASEQ.Time,BASEQ.LSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.RSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.LEAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.REAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.LHAngularWorkonClub);
plot(BASEQ.Time,BASEQ.RHAngularWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Work','RS Angular Work','LE Angular Work','RE Angular Work','LH Angular Work','RH Angular Work');
legend('Location','southeast');
%Add a Title
title('Angular Work on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Angular Work');
pause(PauseTime);
%Close Figure
close(101);

%% ====== End of SCRIPT_101_3D_PLOT_BaseData_AngularWork.m ======

%% ====== Start of SCRIPT_102_3D_PLOT_BaseData_AngularPower.m ======
figure(102);
hold on;
plot(BASEQ.Time,BASEQ.LSonArmAngularPower);
plot(BASEQ.Time,BASEQ.RSonArmAngularPower);
plot(BASEQ.Time,BASEQ.LEonForearmAngularPower);
plot(BASEQ.Time,BASEQ.REonForearmAngularPower);
plot(BASEQ.Time,BASEQ.LHonClubAngularPower);
plot(BASEQ.Time,BASEQ.RHonClubAngularPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Power','RS Angular Power','LE Angular Power','RE Angular Power','LH Angular Power','RH Angular Power');
legend('Location','southeast');
%Add a Title
title('Angular Power on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Angular Power');
pause(PauseTime);
%Close Figure
close(102);

%% ====== End of SCRIPT_102_3D_PLOT_BaseData_AngularPower.m ======

%% ====== Start of SCRIPT_103_3D_PLOT_BaseData_LinearPower.m ======
figure(103);
hold on;
plot(BASEQ.Time,BASEQ.LSonArmLinearPower);
plot(BASEQ.Time,BASEQ.RSonArmLinearPower);
plot(BASEQ.Time,BASEQ.LEonForearmLinearPower);
plot(BASEQ.Time,BASEQ.REonForearmLinearPower);
plot(BASEQ.Time,BASEQ.LHonClubLinearPower);
plot(BASEQ.Time,BASEQ.RHonClubLinearPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Power','RS Linear Power','LE Linear Power','RE Linear Power','LH Linear Power','RH Linear Power');
legend('Location','southeast');
%Add a Title
title('Linear Power on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Linear Power');
pause(PauseTime);
%Close Figure
close(103);

%% ====== End of SCRIPT_103_3D_PLOT_BaseData_LinearPower.m ======

%% ====== Start of SCRIPT_104_3D_PLOT_BaseData_LinearWork.m ======
figure(104);
hold on;
plot(BASEQ.Time,BASEQ.LSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.RSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.LELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.RELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Work','RS Linear Work','LE Linear Work','RE Linear Work','LH Linear Work','RH Linear Work');
legend('Location','southeast');
%Add a Title
title('Linear Work on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Linear Work on Distal');
pause(PauseTime);
%Close Figure
close(104);

%% ====== End of SCRIPT_104_3D_PLOT_BaseData_LinearWork.m ======

%% ====== Start of SCRIPT_105_3D_PLOT_BaseData_JointTorqueInputs.m ======
figure(105);
hold on;
plot(BASEQ.Time,BASEQ.HipTorqueXInput);
plot(BASEQ.Time,BASEQ.HipTorqueYInput);
plot(BASEQ.Time,BASEQ.HipTorqueZInput);
plot(BASEQ.Time,BASEQ.TranslationForceXInput);
plot(BASEQ.Time,BASEQ.TranslationForceYInput);
plot(BASEQ.Time,BASEQ.TranslationForceZInput);
plot(BASEQ.Time,BASEQ.TorsoTorqueInput);
plot(BASEQ.Time,BASEQ.SpineTorqueXInput);
plot(BASEQ.Time,BASEQ.SpineTorqueYInput);
plot(BASEQ.Time,BASEQ.LScapTorqueXInput);
plot(BASEQ.Time,BASEQ.LScapTorqueYInput);
plot(BASEQ.Time,BASEQ.RScapTorqueXInput);
plot(BASEQ.Time,BASEQ.RScapTorqueYInput);
plot(BASEQ.Time,BASEQ.LSTorqueXInput);
plot(BASEQ.Time,BASEQ.LSTorqueYInput);
plot(BASEQ.Time,BASEQ.LSTorqueZInput);
plot(BASEQ.Time,BASEQ.RSTorqueXInput);
plot(BASEQ.Time,BASEQ.RSTorqueYInput);
plot(BASEQ.Time,BASEQ.RSTorqueZInput);
plot(BASEQ.Time,BASEQ.LETorqueInput);
plot(BASEQ.Time,BASEQ.RETorqueInput);
plot(BASEQ.Time,BASEQ.LFTorqueInput);
plot(BASEQ.Time,BASEQ.RFTorqueInput);
plot(BASEQ.Time,BASEQ.LWTorqueXInput);
plot(BASEQ.Time,BASEQ.LWTorqueYInput);
plot(BASEQ.Time,BASEQ.RWTorqueXInput);
plot(BASEQ.Time,BASEQ.RWTorqueYInput);
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
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Joint Torque Inputs');
pause(PauseTime);
%Close Figure
close(105);

%% ====== End of SCRIPT_105_3D_PLOT_BaseData_JointTorqueInputs.m ======

%% ====== Start of SCRIPT_106_3D_PLOT_BaseData_TotalWork.m ======
figure(106);
hold on;
plot(BASEQ.Time,BASEQ.TotalLSWork);
plot(BASEQ.Time,BASEQ.TotalRSWork);
plot(BASEQ.Time,BASEQ.TotalLEWork);
plot(BASEQ.Time,BASEQ.TotalREWork);
plot(BASEQ.Time,BASEQ.TotalLHWork);
plot(BASEQ.Time,BASEQ.TotalRHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Total Work','RS Total Work','LE Total Work','RE Total Work','LH Total Work','RH Total Work');
legend('Location','southeast');
%Add a Title
title('Total Work on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Total Work');
pause(PauseTime);
%Close Figure
close(106);

%% ====== End of SCRIPT_106_3D_PLOT_BaseData_TotalWork.m ======

%% ====== Start of SCRIPT_107_3D_PLOT_BaseData_TotalPower.m ======
figure(107);
hold on;
plot(BASEQ.Time,BASEQ.TotalLSPower);
plot(BASEQ.Time,BASEQ.TotalRSPower);
plot(BASEQ.Time,BASEQ.TotalLEPower);
plot(BASEQ.Time,BASEQ.TotalREPower);
plot(BASEQ.Time,BASEQ.TotalLHPower);
plot(BASEQ.Time,BASEQ.TotalRHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Total Power','RS Total Power','LE Total Power','RE Total Power','LW Total Power','RW Total Power');
legend('Location','southeast');
%Add a Title
title('Total Power on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Total Power');
pause(PauseTime);
%Close Figure
close(107);

%% ====== End of SCRIPT_107_3D_PLOT_BaseData_TotalPower.m ======

%% ====== Start of SCRIPT_122_3D_PLOT_BaseData_ForceAlongHandPath.m ======
figure(122);
plot(BASEQ.Time,BASEQ.ForceAlongHandPath);
xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('Force Along Hand Path');
legend('Location','southeast');
%Add a Title
title('Force Along Hand Path');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Force Along Hand Path');
pause(PauseTime);
%Close Figure
close(122);

%% ====== End of SCRIPT_122_3D_PLOT_BaseData_ForceAlongHandPath.m ======

%% ====== Start of SCRIPT_123_3D_PLOT_BaseData_CHSandHandSpeed.m ======
figure(123);
hold on;
plot(BASEQ.Time,BASEQ.("CHS (mph)"));
plot(BASEQ.Time,BASEQ.("Hand Speed (mph)"));
xlabel('Time (s)');
ylabel('Speed (mph)');
grid 'on';
%Add Legend to Plot
legend('Clubhead Speed (mph)','Hand Speed (mph)');
legend('Location','southeast');
%Add a Title
title('Clubhead and Hand Speed');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - CHS and Hand Speed');
pause(PauseTime);
%Close Figure
close(123);

%% ====== End of SCRIPT_123_3D_PLOT_BaseData_CHSandHandSpeed.m ======

%% ====== Start of SCRIPT_125_3D_PLOT_BaseData_LHRHForceAlongHandPath.m ======
figure(125);
hold on;
plot(BASEQ.Time,BASEQ.LHForceAlongHandPath);
plot(BASEQ.Time,BASEQ.RHForceAlongHandPath);
plot(BASEQ.Time,BASEQ.ForceAlongHandPath);
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('LH Force on Left Hand Path','RH Force on Right Hand Path','Net Force Along MP Hand Path');
legend('Location','southeast');
%Add a Title
title('Force Along Hand Path');
%subtitle('Left Hand, Right Hand, Total');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Force Along Hand Path - LH RH Total');
pause(PauseTime);
%Close Figure
close(125);

%% ====== End of SCRIPT_125_3D_PLOT_BaseData_LHRHForceAlongHandPath.m ======

%% ====== Start of SCRIPT_126_3D_PLOT_BaseData_LinearImpulse.m ======
figure(126);
plot(BASEQ.Time,BASEQ.LinearImpulseonClub);
xlabel('Time (s)');
ylabel('Impulse (Ns)');
grid 'on';
%Add Legend to Plot
legend('Linear Impulse');
legend('Location','southeast');
%Add a Title
title('Linear Impulse');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Linear Impulse');
pause(PauseTime);
%Close Figure
close(126);

%% ====== End of SCRIPT_126_3D_PLOT_BaseData_LinearImpulse.m ======

%% ====== Start of SCRIPT_127_3D_PLOT_BaseData_LinearWork.m ======
figure(127);
hold on;
plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.LinearWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Work','RH Linear Work','Net Force Linear Work (midpoint)');
legend('Location','southeast');
%Add a Title
title('Linear Work');
subtitle('Left Hand, Right Hand, Total');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Linear Work on Club');
subtitle('BaseData');
pause(PauseTime);
%Close Figure
close(127);

%% ====== End of SCRIPT_127_3D_PLOT_BaseData_LinearWork.m ======

%% ====== Start of SCRIPT_128_3D_PLOT_BaseData_LinearImpulse.m ======
figure(128);
hold on;
plot(BASEQ.Time,BASEQ.LHLinearImpulseonClub);
plot(BASEQ.Time,BASEQ.RHLinearImpulseonClub);
plot(BASEQ.Time,BASEQ.LinearImpulseonClub);
ylabel('Linear Impulse (kgm/s)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Impulse','RH Linear Impulse','Net Force Linear Impulse (midpoint)');
legend('Location','southeast');
%Add a Title
title('Linear Impulse');
%subtitle('Left Hand, Right Hand, Total');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Linear Impulse LH,RH,Total');
pause(PauseTime);
%Close Figure
close(128);

%% ====== End of SCRIPT_128_3D_PLOT_BaseData_LinearImpulse.m ======

%% ====== Start of SCRIPT_129_3D_PLOT_BaseData_EquivalentCoupleandMOF.m ======
figure(129);
hold on;
EQMPLOCAL=BASEQ.EquivalentMidpointCoupleLocal(:,3);
MPMOFLOCAL=BASEQ.MPMOFonClubLocal(:,3);
SUMOFMOMENTSLOCAL=BASEQ.SumofMomentsonClubLocal(:,3);
plot(BASEQ.Time,EQMPLOCAL);
plot(BASEQ.Time,MPMOFLOCAL);
plot(BASEQ.Time,SUMOFMOMENTSLOCAL);
ylabel('Torque (Nm)');
grid 'on';
%Add Legend to Plot
legend('Equivalent Midpoint Couple','Total Force on Midpoint MOF','Sum of Moments');
legend('Location','southeast');
%Add a Title
title('Equivalent Couple, Moment of Force, Sum of Moments');
subtitle('BASE - Grip Reference Frame');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Equivalent Couple and MOF');
pause(PauseTime);
%Close Figure
close(129);

%% ====== End of SCRIPT_129_3D_PLOT_BaseData_EquivalentCoupleandMOF.m ======

%% ====== Start of SCRIPT_130_3D_PLOT_BaseData_LSWork.m ======
figure(130);
hold on;
plot(BASEQ.Time,BASEQ.LSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.LSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.TotalLSWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Work','LS Angular Work','LS Total Work');
legend('Location','southeast');
%Add a Title
title('Left Shoulder Work on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Shoulder Work');
pause(PauseTime);
%Close Figure
close(130);

%% ====== End of SCRIPT_130_3D_PLOT_BaseData_LSWork.m ======

%% ====== Start of SCRIPT_131_3D_PLOT_BaseData_LSPower.m ======
figure(131);
hold on;
plot(BASEQ.Time,BASEQ.LSonArmLinearPower);
plot(BASEQ.Time,BASEQ.LSonArmAngularPower);
plot(BASEQ.Time,BASEQ.TotalLSPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Power','LS Angular Power','LS Total Power');
legend('Location','southeast');
%Add a Title
title('Left Shoulder Power on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Shoulder Power');
pause(PauseTime);
%Close Figure
close(131);

%% ====== End of SCRIPT_131_3D_PLOT_BaseData_LSPower.m ======

%% ====== Start of SCRIPT_132_3D_PLOT_BaseData_RSWork.m ======
figure(132);
hold on;
plot(BASEQ.Time,BASEQ.RSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.RSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.TotalRSWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RS Linear Work','RS Angular Work','RS Total Work');
legend('Location','southeast');
%Add a Title
title('Right Shoulder Work on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Shoulder Work');
pause(PauseTime);
%Close Figure
close(132);

%% ====== End of SCRIPT_132_3D_PLOT_BaseData_RSWork.m ======

%% ====== Start of SCRIPT_133_3D_PLOT_BaseData_RSPower.m ======
figure(133);
hold on;
plot(BASEQ.Time,BASEQ.RSonArmLinearPower);
plot(BASEQ.Time,BASEQ.RSonArmAngularPower);
plot(BASEQ.Time,BASEQ.TotalRSPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RS Linear Power','RS Angular Power','RS Total Power');
legend('Location','southeast');
%Add a Title
title('Right Shoulder Power on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Shoulder Power');
pause(PauseTime);
%Close Figure
close(133);

%% ====== End of SCRIPT_133_3D_PLOT_BaseData_RSPower.m ======

%% ====== Start of SCRIPT_134_3D_PLOT_BaseData_LEWork.m ======
figure(134);
hold on;
plot(BASEQ.Time,BASEQ.LELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.LEAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.TotalLEWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LE Linear Work','LE Angular Work','LE Total Work');
legend('Location','southeast');
%Add a Title
title('Left Elbow Work on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Elbow Work');
pause(PauseTime);
%Close Figure
close(134);

%% ====== End of SCRIPT_134_3D_PLOT_BaseData_LEWork.m ======

%% ====== Start of SCRIPT_135_3D_PLOT_BaseData_LEPower.m ======
figure(135);
hold on;
plot(BASEQ.Time,BASEQ.LEonForearmLinearPower);
plot(BASEQ.Time,BASEQ.LEonForearmAngularPower);
plot(BASEQ.Time,BASEQ.TotalLEPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LE Linear Power','LE Angular Power','LE Total Power');
legend('Location','southeast');
%Add a Title
title('Left Elbow Power on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Elbow Power');
pause(PauseTime);
%Close Figure
close(135);

%% ====== End of SCRIPT_135_3D_PLOT_BaseData_LEPower.m ======

%% ====== Start of SCRIPT_136_3D_PLOT_BaseData_REWork.m ======
figure(136);
hold on;
plot(BASEQ.Time,BASEQ.RELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.REAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.TotalREWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RE Linear Work','RE Angular Work','RE Total Work');
legend('Location','southeast');
%Add a Title
title('Right Elbow Work on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Elbow Work');
pause(PauseTime);
%Close Figure
close(136);

%% ====== End of SCRIPT_136_3D_PLOT_BaseData_REWork.m ======

%% ====== Start of SCRIPT_137_3D_PLOT_BaseData_REPower.m ======
figure(137);
hold on;
plot(BASEQ.Time,BASEQ.REonForearmLinearPower);
plot(BASEQ.Time,BASEQ.REonForearmAngularPower);
plot(BASEQ.Time,BASEQ.TotalREPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RE Linear Power','RE Angular Power','RE Total Power');
legend('Location','southeast');
%Add a Title
title('Right Elbow Power on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Elbow Power');
pause(PauseTime);
%Close Figure
close(137);

%% ====== End of SCRIPT_137_3D_PLOT_BaseData_REPower.m ======

%% ====== Start of SCRIPT_138_3D_PLOT_BaseData_LHWork.m ======
figure(138);
hold on;
plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.LHAngularWorkonClub);
plot(BASEQ.Time,BASEQ.TotalLHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Work','LH Angular Work','LH Total Work');
legend('Location','southeast');
%Add a Title
title('Left Wrist Work on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Wrist Work');
pause(PauseTime);
%Close Figure
close(138);

%% ====== End of SCRIPT_138_3D_PLOT_BaseData_LHWork.m ======

%% ====== Start of SCRIPT_139_3D_PLOT_BaseData_LHPower.m ======
figure(139);
hold on;
plot(BASEQ.Time,BASEQ.LHonClubLinearPower);
plot(BASEQ.Time,BASEQ.LHonClubAngularPower);
plot(BASEQ.Time,BASEQ.TotalLHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Power','LH Angular Power','LH Total Power');
legend('Location','southeast');
%Add a Title
title('Left Wrist Power on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Wrist Power');
pause(PauseTime);
%Close Figure
close(139);

%% ====== End of SCRIPT_139_3D_PLOT_BaseData_LHPower.m ======

%% ====== Start of SCRIPT_140_3D_PLOT_BaseData_RHWork.m ======
figure(140);
hold on;
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHAngularWorkonClub);
plot(BASEQ.Time,BASEQ.TotalRHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RH Linear Work','RH Angular Work','RH Total Work');
legend('Location','southeast');
%Add a Title
title('Right Wrist Work on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Wrist Work');
pause(PauseTime);
%Close Figure
close(140);

%% ====== End of SCRIPT_140_3D_PLOT_BaseData_RHWork.m ======

%% ====== Start of SCRIPT_141_3D_PLOT_BaseData_RHPower.m ======
figure(141);
hold on;
plot(BASEQ.Time,BASEQ.RHonClubLinearPower);
plot(BASEQ.Time,BASEQ.RHonClubAngularPower);
plot(BASEQ.Time,BASEQ.TotalRHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RH Linear Power','RH Angular Power','RH Total Power');
legend('Location','southeast');
%Add a Title
title('Right Wrist Power on Distal Segment');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Wrist Power');
pause(PauseTime);
%Close Figure
close(141);

%% ====== End of SCRIPT_141_3D_PLOT_BaseData_RHPower.m ======

%% ====== Start of SCRIPT_148_3D_PLOT_BaseData_LocalHandForces.m ======
figure(148);
hold on;
plot(BASEQ.Time(:,1),BASEQ.LHonClubForceLocal(:,1));
plot(BASEQ.Time(:,1),BASEQ.LHonClubForceLocal(:,2));
plot(BASEQ.Time(:,1),BASEQ.LHonClubForceLocal(:,3));
plot(BASEQ.Time(:,1),BASEQ.RHonClubForceLocal(:,1));
plot(BASEQ.Time(:,1),BASEQ.RHonClubForceLocal(:,2));
plot(BASEQ.Time(:,1),BASEQ.RHonClubForceLocal(:,3));
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('Left Hand X','Left Hand Y','Left Hand Z','Right Hand X','Right Hand Y','Right Hand Z');
legend('Location','southeast');
%Add a Title
title('Local Hand Forces on Club');
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Local Hand Forces');
pause(PauseTime);
%Close Figure
close(148);

%% ====== End of SCRIPT_148_3D_PLOT_BaseData_LocalHandForces.m ======

%% ====== Start of SCRIPT_150_3D_PLOT_BaseData_TorqueInputvsTorqueOutput.m ======
figure(150);
hold on;
plot(BASEQ.Time,BASEQ.HipConstraintTorqueX);
plot(BASEQ.Time,BASEQ.HipConstraintTorqueY);
plot(BASEQ.Time,BASEQ.HipConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.TorsoConstraintTorqueX);
plot(BASEQ.Time,BASEQ.TorsoConstraintTorqueY);
plot(BASEQ.Time,BASEQ.TorsoConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.SpineConstraintTorqueX);
plot(BASEQ.Time,BASEQ.SpineConstraintTorqueY);
plot(BASEQ.Time,BASEQ.SpineConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.LScapConstraintTorqueX);
plot(BASEQ.Time,BASEQ.LScapConstraintTorqueY);
plot(BASEQ.Time,BASEQ.LScapConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.RScapConstraintTorqueX);
plot(BASEQ.Time,BASEQ.RScapConstraintTorqueY);
plot(BASEQ.Time,BASEQ.RScapConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.LSConstraintTorqueX);
plot(BASEQ.Time,BASEQ.LSConstraintTorqueY);
plot(BASEQ.Time,BASEQ.LSConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.RSConstraintTorqueX);
plot(BASEQ.Time,BASEQ.RSConstraintTorqueY);
plot(BASEQ.Time,BASEQ.RSConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.LEConstraintTorqueX);
plot(BASEQ.Time,BASEQ.LEConstraintTorqueY);
plot(BASEQ.Time,BASEQ.LEConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.REConstraintTorqueX);
plot(BASEQ.Time,BASEQ.REConstraintTorqueY);
plot(BASEQ.Time,BASEQ.REConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.LFConstraintTorqueX);
plot(BASEQ.Time,BASEQ.LFConstraintTorqueY);
plot(BASEQ.Time,BASEQ.LFConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.RFConstraintTorqueX);
plot(BASEQ.Time,BASEQ.RFConstraintTorqueY);
plot(BASEQ.Time,BASEQ.RFConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.LWConstraintTorqueX);
plot(BASEQ.Time,BASEQ.LWConstraintTorqueY);
plot(BASEQ.Time,BASEQ.LWConstraintTorqueZ);
plot(BASEQ.Time,BASEQ.RWConstraintTorqueX);
plot(BASEQ.Time,BASEQ.RWConstraintTorqueY);
plot(BASEQ.Time,BASEQ.RWConstraintTorqueZ);
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
subtitle('BASE');
%Save Figure
savefig('BaseData Charts/BASE_Plot - Joint Torque Inputs');
pause(PauseTime);
%Close Figure
close(150);

%% ====== End of SCRIPT_150_3D_PLOT_BaseData_TorqueInputvsTorqueOutput.m ======

end
