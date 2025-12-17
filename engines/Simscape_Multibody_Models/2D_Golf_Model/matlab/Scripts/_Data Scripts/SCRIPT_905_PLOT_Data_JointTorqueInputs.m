figure(905);
hold on;

plot(Data.Time,Data.HipTorque(:,3));
plot(Data.Time,Data.TorsoTorque(:,3));
plot(Data.Time,Data.LScapTorque(:,3));
plot(Data.Time,Data.RScapTorque(:,3));
plot(Data.Time,Data.LSTorqueLocal(:,3));
plot(Data.Time,Data.RSTorqueLocal(:,3));
plot(Data.Time,Data.LeftElbowTorqueLocal(:,3));
plot(Data.Time,Data.RightElbowTorque(:,3));
plot(Data.Time,Data.LeftWristTorqueLocal(:,3));
plot(Data.Time,Data.RightWristTorqueLocal(:,3));

ylabel('Torque (Nm)');
grid 'on';

%Add Legend to Plot
legend('Hip Torque','Torso Torque', 'Left Scap Torque','Right Scap Torque','Left Shoulder Torque','Right Shoulder Torque','Left Elbow Torque','Right Elbow Torque','Left Wrist Torque','Right Wrist Torque');
legend('Location','southeast');

%Add a Title
title('Joint Torque Inputs');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Joint Torque Inputs');
pause(PauseTime);

%Close Figure
close(905);