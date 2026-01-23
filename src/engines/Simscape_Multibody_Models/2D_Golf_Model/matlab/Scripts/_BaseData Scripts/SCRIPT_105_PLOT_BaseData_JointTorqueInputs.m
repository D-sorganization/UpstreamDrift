figure(105);
hold on;

plot(BASEQ.Time,BASEQ.HipTorque(:,3));
plot(BASEQ.Time,BASEQ.TorsoTorque(:,3));
plot(BASEQ.Time,BASEQ.LScapTorque(:,3));
plot(BASEQ.Time,BASEQ.RScapTorque(:,3));
plot(BASEQ.Time,BASEQ.LSTorqueLocal(:,3));
plot(BASEQ.Time,BASEQ.RSTorqueLocal(:,3));
plot(BASEQ.Time,BASEQ.LeftElbowTorqueLocal(:,3));
plot(BASEQ.Time,BASEQ.RightElbowTorque(:,3));
plot(BASEQ.Time,BASEQ.LeftWristTorqueLocal(:,3));
plot(BASEQ.Time,BASEQ.RightWristTorqueLocal(:,3));

ylabel('Torque (Nm)');
grid 'on';

%Add Legend to Plot
legend('Hip Torque','Torso Torque', 'Left Scap Torque','Right Scap Torque','Left Shoulder Torque','Right Shoulder Torque','Left Elbow Torque','Right Elbow Torque','Left Wrist Torque','Right Wrist Torque');
legend('Location','southeast');

%Add a Title
title('Joint Torque Inputs');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Joint Torque Inputs');
pause(PauseTime);

%Close Figure
close(105);