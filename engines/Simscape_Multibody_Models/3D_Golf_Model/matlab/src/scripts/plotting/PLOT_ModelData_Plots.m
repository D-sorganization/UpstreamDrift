function PLOT__Plots()
%%% ====== Start of SCRIPT_901_3D_PLOT_Data_AngularWork.m ======
figure(901);
hold on;
plot(Data.Time,Data.LSAngularWorkonArm);
plot(Data.Time,Data.RSAngularWorkonArm);
plot(Data.Time,Data.LEAngularWorkonForearm);
plot(Data.Time,Data.REAngularWorkonForearm);
plot(Data.Time,Data.LHAngularWorkonClub);
plot(Data.Time,Data.RHAngularWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Work','RS Angular Work','LE Angular Work','RE Angular Work','LH Angular Work','RH Angular Work');
legend('Location','southeast');
%Add a Title
title('Angular Work on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Angular Work');
pause(PauseTime);
%Close Figure
close(901);

%% ====== End of SCRIPT_901_3D_PLOT_Data_AngularWork.m ======

%% ====== Start of SCRIPT_902_3D_PLOT_Data_AngularPower.m ======
figure(902);
hold on;
plot(Data.Time,Data.LSonArmAngularPower);
plot(Data.Time,Data.RSonArmAngularPower);
plot(Data.Time,Data.LEonForearmAngularPower);
plot(Data.Time,Data.REonForearmAngularPower);
plot(Data.Time,Data.LHonClubAngularPower);
plot(Data.Time,Data.RHonClubAngularPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Angular Power','RS Angular Power','LE Angular Power','RE Angular Power','LH Angular Power','RH Angular Power');
legend('Location','southeast');
%Add a Title
title('Angular Power on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Angular Power');
pause(PauseTime);
%Close Figure
close(902);

%% ====== End of SCRIPT_902_3D_PLOT_Data_AngularPower.m ======

%% ====== Start of SCRIPT_903_3D_PLOT_Data_LinearPower.m ======
figure(903);
hold on;
plot(Data.Time,Data.LSonArmLinearPower);
plot(Data.Time,Data.RSonArmLinearPower);
plot(Data.Time,Data.LEonForearmLinearPower);
plot(Data.Time,Data.REonForearmLinearPower);
plot(Data.Time,Data.LHonClubLinearPower);
plot(Data.Time,Data.RHonClubLinearPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Power','RS Linear Power','LE Linear Power','RE Linear Power','LH Linear Power','RH Linear Power');
legend('Location','southeast');
%Add a Title
title('Linear Power on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Linear Power');
pause(PauseTime);
%Close Figure
close(903);

%% ====== End of SCRIPT_903_3D_PLOT_Data_LinearPower.m ======

%% ====== Start of SCRIPT_904_3D_PLOT_Data_LinearWork.m ======
figure(904);
hold on;
plot(Data.Time,Data.LSLinearWorkonArm);
plot(Data.Time,Data.RSLinearWorkonArm);
plot(Data.Time,Data.LELinearWorkonForearm);
plot(Data.Time,Data.RELinearWorkonForearm);
plot(Data.Time,Data.LHLinearWorkonClub);
plot(Data.Time,Data.RHLinearWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Work','RS Linear Work','LE Linear Work','RE Linear Work','LW Linear Work','RW Linear Work');
legend('Location','southeast');
%Add a Title
title('Linear Work on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Linear Work on Distal');
pause(PauseTime);
%Close Figure
close(904);

%% ====== End of SCRIPT_904_3D_PLOT_Data_LinearWork.m ======

%% ====== Start of SCRIPT_905_3D_PLOT_Data_JointTorqueInputs.m ======
figure(905);
hold on;
plot(Data.Time,Data.HipTorqueXInput);
plot(Data.Time,Data.HipTorqueYInput);
plot(Data.Time,Data.HipTorqueZInput);
plot(Data.Time,Data.TranslationForceXInput);
plot(Data.Time,Data.TranslationForceYInput);
plot(Data.Time,Data.TranslationForceZInput);
plot(Data.Time,Data.TorsoTorqueInput);
plot(Data.Time,Data.SpineTorqueXInput);
plot(Data.Time,Data.SpineTorqueYInput);
plot(Data.Time,Data.LScapTorqueXInput);
plot(Data.Time,Data.LScapTorqueYInput);
plot(Data.Time,Data.RScapTorqueXInput);
plot(Data.Time,Data.RScapTorqueYInput);
plot(Data.Time,Data.LSTorqueXInput);
plot(Data.Time,Data.LSTorqueYInput);
plot(Data.Time,Data.LSTorqueZInput);
plot(Data.Time,Data.RSTorqueXInput);
plot(Data.Time,Data.RSTorqueYInput);
plot(Data.Time,Data.RSTorqueZInput);
plot(Data.Time,Data.LETorqueInput);
plot(Data.Time,Data.RETorqueInput);
plot(Data.Time,Data.LFTorqueInput);
plot(Data.Time,Data.RFTorqueInput);
plot(Data.Time,Data.LWTorqueXInput);
plot(Data.Time,Data.LWTorqueYInput);
plot(Data.Time,Data.RWTorqueXInput);
plot(Data.Time,Data.RWTorqueYInput);
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
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Joint Torque Inputs');
pause(PauseTime);
%Close Figure
close(905);

%% ====== End of SCRIPT_905_3D_PLOT_Data_JointTorqueInputs.m ======

%% ====== Start of SCRIPT_906_3D_PLOT_Data_TotalWork.m ======
figure(906);
hold on;
plot(Data.Time,Data.TotalLSWork);
plot(Data.Time,Data.TotalRSWork);
plot(Data.Time,Data.TotalLEWork);
plot(Data.Time,Data.TotalREWork);
plot(Data.Time,Data.TotalLHWork);
plot(Data.Time,Data.TotalRHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Total Work','RS Total Work','LE Total Work','RE Total Work','LW Total Work','RW Total Work');
legend('Location','southeast');
%Add a Title
title('Total Work on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Total Work');
pause(PauseTime);
%Close Figure
close(906);

%% ====== End of SCRIPT_906_3D_PLOT_Data_TotalWork.m ======

%% ====== Start of SCRIPT_907_3D_PLOT_Data_TotalPower.m ======
figure(907);
hold on;
plot(Data.Time,Data.TotalLSPower);
plot(Data.Time,Data.TotalRSPower);
plot(Data.Time,Data.TotalLEPower);
plot(Data.Time,Data.TotalREPower);
plot(Data.Time,Data.TotalLHPower);
plot(Data.Time,Data.TotalRHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Total Power','RS Total Power','LE Total Power','RE Total Power','LH Total Power','RH Total Power');
legend('Location','southeast');
%Add a Title
title('Total Power on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Total Power');
pause(PauseTime);
%Close Figure
close(907);

%% ====== End of SCRIPT_907_3D_PLOT_Data_TotalPower.m ======

%% ====== Start of SCRIPT_922_3D_PLOT_Data_ForceAlongHandPath.m ======
figure(922);
plot(Data.Time,Data.ForceAlongHandPath);
xlabel('Time (s)');
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('Force Along Hand Path');
legend('Location','southeast');
%Add a Title
title('Force Along Hand Path');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Plot - Force Along Hand Path');
pause(PauseTime);
%Close Figure
close(922);

%% ====== End of SCRIPT_922_3D_PLOT_Data_ForceAlongHandPath.m ======

%% ====== Start of SCRIPT_923_3D_PLOT_Data_CHSandHandSpeed.m ======
figure(923);
hold on;
plot(Data.Time,Data.("CHS (mph)"));
plot(Data.Time,Data.("Hand Speed (mph)"));
xlabel('Time (s)');
ylabel('Speed (mph)');
grid 'on';
%Add Legend to Plot
legend('Clubhead Speed (mph)','Hand Speed (mph)');
legend('Location','southeast');
%Add a Title
title('Clubhead and Hand Speed');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Plot - CHS and Hand Speed');
pause(PauseTime);
%Close Figure
close(923);

%% ====== End of SCRIPT_923_3D_PLOT_Data_CHSandHandSpeed.m ======

%% ====== Start of SCRIPT_925_3D_PLOT_Data_LHRHForceAlongHandPath.m ======
figure(925);
hold on;
plot(Data.Time,Data.LHForceAlongHandPath);
plot(Data.Time,Data.RHForceAlongHandPath);
plot(Data.Time,Data.ForceAlongHandPath);
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('LH Force on Left Hand Path','RH Force on Right Hand Path','Net Force Along MP Hand Path');
legend('Location','southeast');
%Add a Title
title('Force Along Hand Path');
subtitle('Data');
%subtitle('Left Hand, Right Hand, Total');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Plot - Force Along Hand Path - LH RH Total');
pause(PauseTime);
%Close Figure
close(925);

%% ====== End of SCRIPT_925_3D_PLOT_Data_LHRHForceAlongHandPath.m ======

%% ====== Start of SCRIPT_926_3D_PLOT_Data_LinearImpulse.m ======
figure(926);
plot(Data.Time,Data.LinearImpulseonClub);
xlabel('Time (s)');
ylabel('Impulse (Ns)');
grid 'on';
%Add Legend to Plot
legend('Linear Impulse');
legend('Location','southeast');
%Add a Title
title('Linear Impulse');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Plot - Linear Impulse');
pause(PauseTime);
%Close Figure
close(926);

%% ====== End of SCRIPT_926_3D_PLOT_Data_LinearImpulse.m ======

%% ====== Start of SCRIPT_927_3D_PLOT_Data_LinearWork.m ======
figure(927);
hold on;
plot(Data.Time,Data.LHLinearWorkonClub);
plot(Data.Time,Data.RHLinearWorkonClub);
plot(Data.Time,Data.LinearWorkonClub);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Work','RH Linear Work','Net Force Linear Work (midpoint)');
legend('Location','southeast');
%Add a Title
title('Linear Work');
subtitle('Data');
%subtitle('Left Hand, Right Hand, Total');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Plot - Linear Work');
pause(PauseTime);
%Close Figure
close(927);

%% ====== End of SCRIPT_927_3D_PLOT_Data_LinearWork.m ======

%% ====== Start of SCRIPT_928_3D_PLOT_Data_LinearImpulse.m ======
figure(928);
hold on;
plot(Data.Time,Data.LHLinearImpulseonClub);
plot(Data.Time,Data.RHLinearImpulseonClub);
plot(Data.Time,Data.LinearImpulseonClub);
ylabel('Linear Impulse (kgm/s)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Impulse','RH Linear Impulse','Net Force Linear Impulse (midpoint)');
legend('Location','southeast');
%Add a Title
title('Linear Impulse');
subtitle('Data');
%subtitle('Left Hand, Right Hand, Total');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Plot - Linear Impulse LH,RH,Total');
pause(PauseTime);
%Close Figure
close(928);

%% ====== End of SCRIPT_928_3D_PLOT_Data_LinearImpulse.m ======

%% ====== Start of SCRIPT_929_3D_PLOT_Data_EquivalentCoupleandMOF.m ======
figure(929);
hold on;
EQMPLOCAL=Data.EquivalentMidpointCoupleLocal(:,3);
MPMOFLOCAL=Data.MPMOFonClubLocal(:,3);
SUMOFMOMENTSLOCAL=Data.SumofMomentsonClubLocal(:,3);
plot(Data.Time,EQMPLOCAL);
plot(Data.Time,MPMOFLOCAL);
plot(Data.Time,SUMOFMOMENTSLOCAL);
ylabel('Torque (Nm)');
grid 'on';
%Add Legend to Plot
legend('Equivalent Midpoint Couple','Total Force on Midpoint MOF','Sum of Moments');
legend('Location','southeast');
%Add a Title
title('Equivalent Couple, Moment of Force, Sum of Moments');
subtitle('Data - Grip Reference Frame');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Plot - Equivalent Couple and MOF');
pause(PauseTime);
%Close Figure
close(929);

%% ====== End of SCRIPT_929_3D_PLOT_Data_EquivalentCoupleandMOF.m ======

%% ====== Start of SCRIPT_930_3D_PLOT_Data_LSWork.m ======
figure(930);
hold on;
plot(Data.Time,Data.LSLinearWorkonArm);
plot(Data.Time,Data.LSAngularWorkonArm);
plot(Data.Time,Data.TotalLSWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Work','LS Angular Work','LS Total Work');
legend('Location','southeast');
%Add a Title
title('Left Shoulder Work on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Left Shoulder Work');
pause(PauseTime);
%Close Figure
close(930);

%% ====== End of SCRIPT_930_3D_PLOT_Data_LSWork.m ======

%% ====== Start of SCRIPT_931_3D_PLOT_Data_LSPower.m ======
figure(931);
hold on;
plot(Data.Time,Data.LSonArmLinearPower);
plot(Data.Time,Data.LSonArmAngularPower);
plot(Data.Time,Data.TotalLSPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LS Linear Power','LS Angular Power','LS Total Power');
legend('Location','southeast');
%Add a Title
title('Left Shoulder Power on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Left Shoulder Power');
pause(PauseTime);
%Close Figure
close(931);

%% ====== End of SCRIPT_931_3D_PLOT_Data_LSPower.m ======

%% ====== Start of SCRIPT_932_3D_PLOT_Data_RSWork.m ======
figure(932);
hold on;
plot(Data.Time,Data.RSLinearWorkonArm);
plot(Data.Time,Data.RSAngularWorkonArm);
plot(Data.Time,Data.TotalRSWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RS Linear Work','RS Angular Work','RS Total Work');
legend('Location','southeast');
%Add a Title
title('Right Shoulder Work on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Right Shoulder Work');
pause(PauseTime);
%Close Figure
close(932);

%% ====== End of SCRIPT_932_3D_PLOT_Data_RSWork.m ======

%% ====== Start of SCRIPT_933_3D_PLOT_Data_RSPower.m ======
figure(933);
hold on;
plot(Data.Time,Data.RSonArmLinearPower);
plot(Data.Time,Data.RSonArmAngularPower);
plot(Data.Time,Data.TotalRSPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RS Linear Power','RS Angular Power','RS Total Power');
legend('Location','southeast');
%Add a Title
title('Right Shoulder Power on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Right Shoulder Power');
pause(PauseTime);
%Close Figure
close(933);

%% ====== End of SCRIPT_933_3D_PLOT_Data_RSPower.m ======

%% ====== Start of SCRIPT_934_3D_PLOT_Data_LEWork.m ======
figure(934);
hold on;
plot(Data.Time,Data.LELinearWorkonForearm);
plot(Data.Time,Data.LEAngularWorkonForearm);
plot(Data.Time,Data.TotalLEWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LE Linear Work','LE Angular Work','LE Total Work');
legend('Location','southeast');
%Add a Title
title('Left Elbow Work on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Left Elbow Work');
pause(PauseTime);
%Close Figure
close(934);

%% ====== End of SCRIPT_934_3D_PLOT_Data_LEWork.m ======

%% ====== Start of SCRIPT_935_3D_PLOT_Data_LEPower.m ======
figure(935);
hold on;
plot(Data.Time,Data.LEonForearmLinearPower);
plot(Data.Time,Data.LEonForearmAngularPower);
plot(Data.Time,Data.TotalLEPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LE Linear Power','LE Angular Power','LE Total Power');
legend('Location','southeast');
%Add a Title
title('Left Elbow Power on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Left Elbow Power');
pause(PauseTime);
%Close Figure
close(935);

%% ====== End of SCRIPT_935_3D_PLOT_Data_LEPower.m ======

%% ====== Start of SCRIPT_936_3D_PLOT_Data_REWork.m ======
figure(936);
hold on;
plot(Data.Time,Data.RELinearWorkonForearm);
plot(Data.Time,Data.REAngularWorkonForearm);
plot(Data.Time,Data.TotalREWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RE Linear Work','RE Angular Work','RE Total Work');
legend('Location','southeast');
%Add a Title
title('Right Elbow Work on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Right Elbow Work');
pause(PauseTime);
%Close Figure
close(936);

%% ====== End of SCRIPT_936_3D_PLOT_Data_REWork.m ======

%% ====== Start of SCRIPT_937_3D_PLOT_Data_REPower.m ======
figure(937);
hold on;
plot(Data.Time,Data.REonForearmLinearPower);
plot(Data.Time,Data.REonForearmAngularPower);
plot(Data.Time,Data.TotalREPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RE Linear Power','RE Angular Power','RE Total Power');
legend('Location','southeast');
%Add a Title
title('Right Elbow Power on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Right Elbow Power');
pause(PauseTime);
%Close Figure
close(937);

%% ====== End of SCRIPT_937_3D_PLOT_Data_REPower.m ======

%% ====== Start of SCRIPT_938_3D_PLOT_Data_LHWork.m ======
figure(938);
hold on;
plot(Data.Time,Data.LHLinearWorkonClub);
plot(Data.Time,Data.LHAngularWorkonClub);
plot(Data.Time,Data.TotalLHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Work','LH Angular Work','LH Total Work');
legend('Location','southeast');
%Add a Title
title('Left Wrist Work on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Left Hand Work');
pause(PauseTime);
%Close Figure
close(938);

%% ====== End of SCRIPT_938_3D_PLOT_Data_LHWork.m ======

%% ====== Start of SCRIPT_939_3D_PLOT_Data_LHPower.m ======
figure(939);
hold on;
plot(Data.Time,Data.LHonClubLinearPower);
plot(Data.Time,Data.LHonClubAngularPower);
plot(Data.Time,Data.TotalLHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('LH Linear Power','LH Angular Power','LH Total Power');
legend('Location','southeast');
%Add a Title
title('Left Wrist Power on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Left Wrist Power');
pause(PauseTime);
%Close Figure
close(939);

%% ====== End of SCRIPT_939_3D_PLOT_Data_LHPower.m ======

%% ====== Start of SCRIPT_940_3D_PLOT_Data_RHWork.m ======
figure(940);
hold on;
plot(Data.Time,Data.RHLinearWorkonClub);
plot(Data.Time,Data.RHAngularWorkonClub);
plot(Data.Time,Data.TotalRHWork);
ylabel('Work (J)');
grid 'on';
%Add Legend to Plot
legend('RH Linear Work','RH Angular Work','RH Total Work');
legend('Location','southeast');
%Add a Title
title('Right Wrist Work on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Right Wrist Work');
pause(PauseTime);
%Close Figure
close(940);

%% ====== End of SCRIPT_940_3D_PLOT_Data_RHWork.m ======

%% ====== Start of SCRIPT_941_3D_PLOT_Data_RHPower.m ======
figure(941);
hold on;
plot(Data.Time,Data.RHonClubLinearPower);
plot(Data.Time,Data.RHonClubAngularPower);
plot(Data.Time,Data.TotalRHPower);
ylabel('Power (W)');
grid 'on';
%Add Legend to Plot
legend('RH Linear Power','RH Angular Power','RH Total Power');
legend('Location','southeast');
%Add a Title
title('Right Wrist Power on Distal Segment');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Right Wrist Power');
pause(PauseTime);
%Close Figure
close(941);

%% ====== End of SCRIPT_941_3D_PLOT_Data_RHPower.m ======

%% ====== Start of SCRIPT_948_3D_PLOT_Data_LocalHandForces.m ======
figure(948);
hold on;
plot(Data.Time(:,1),Data.LHonClubForceLocal(:,1));
plot(Data.Time(:,1),Data.LHonClubForceLocal(:,2));
plot(Data.Time(:,1),Data.LHonClubForceLocal(:,3));
plot(Data.Time(:,1),Data.RHonClubForceLocal(:,1));
plot(Data.Time(:,1),Data.RHonClubForceLocal(:,2));
plot(Data.Time(:,1),Data.RHonClubForceLocal(:,3));
ylabel('Force (N)');
grid 'on';
%Add Legend to Plot
legend('Left Wrist X','Left Wrist Y','Left Wrist Z','Right Wrist X','Right Wrist Y','Right Wrist Z');
legend('Location','southeast');
%Add a Title
title('Local Hand Forces on Club');
subtitle('Data');
%Save Figure
cd(matlabdrive);
cd '3DModel';
savefig('Scripts/_Model Data Scripts/Data Charts/Data_Plot - Local Hand Forces');
pause(PauseTime);
%Close Figure
close(948);

%% ====== End of SCRIPT_948_3D_PLOT_Data_LocalHandForces.m ======

%% ====== Start of SCRIPT_950_3D_PLOT_Data_TorqueInputvsTorqueOutput.m ======
figure(950);
hold on;
plot(Data.Time,Data.HipConstraintTorqueX);
plot(Data.Time,Data.HipConstraintTorqueY);
plot(Data.Time,Data.HipConstraintTorqueZ);
plot(Data.Time,Data.TorsoConstraintTorqueX);
plot(Data.Time,Data.TorsoConstraintTorqueY);
plot(Data.Time,Data.TorsoConstraintTorqueZ);
plot(Data.Time,Data.SpineConstraintTorqueX);
plot(Data.Time,Data.SpineConstraintTorqueY);
plot(Data.Time,Data.SpineConstraintTorqueZ);
plot(Data.Time,Data.LScapConstraintTorqueX);
plot(Data.Time,Data.LScapConstraintTorqueY);
plot(Data.Time,Data.LScapConstraintTorqueZ);
plot(Data.Time,Data.RScapConstraintTorqueX);
plot(Data.Time,Data.RScapConstraintTorqueY);
plot(Data.Time,Data.RScapConstraintTorqueZ);
plot(Data.Time,Data.LSConstraintTorqueX);
plot(Data.Time,Data.LSConstraintTorqueY);
plot(Data.Time,Data.LSConstraintTorqueZ);
plot(Data.Time,Data.RSConstraintTorqueX);
plot(Data.Time,Data.RSConstraintTorqueY);
plot(Data.Time,Data.RSConstraintTorqueZ);
plot(Data.Time,Data.LEConstraintTorqueX);
plot(Data.Time,Data.LEConstraintTorqueY);
plot(Data.Time,Data.LEConstraintTorqueZ);
plot(Data.Time,Data.REConstraintTorqueX);
plot(Data.Time,Data.REConstraintTorqueY);
plot(Data.Time,Data.REConstraintTorqueZ);
plot(Data.Time,Data.LFConstraintTorqueX);
plot(Data.Time,Data.LFConstraintTorqueY);
plot(Data.Time,Data.LFConstraintTorqueZ);
plot(Data.Time,Data.RFConstraintTorqueX);
plot(Data.Time,Data.RFConstraintTorqueY);
plot(Data.Time,Data.RFConstraintTorqueZ);
plot(Data.Time,Data.LWConstraintTorqueX);
plot(Data.Time,Data.LWConstraintTorqueY);
plot(Data.Time,Data.LWConstraintTorqueZ);
plot(Data.Time,Data.RWConstraintTorqueX);
plot(Data.Time,Data.RWConstraintTorqueY);
plot(Data.Time,Data.RWConstraintTorqueZ);
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
subtitle('Data');
%Save Figure
cd(matlabdrive);
savefig('3DModel/Scripts/_Model Data Scripts/Data Charts/Data_Plot - Joint Torque Inputs');
pause(PauseTime);
%Close Figure
close(950);

%% ====== End of SCRIPT_950_3D_PLOT_Data_TorqueInputvsTorqueOutput.m ======

end
