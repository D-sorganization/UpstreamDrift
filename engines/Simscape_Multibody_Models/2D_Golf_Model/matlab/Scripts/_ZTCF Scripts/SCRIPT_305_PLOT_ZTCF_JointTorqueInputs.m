figure(305);
hold on;

plot(ZTCFQ.Time,ZTCFQ.HipTorque(:,3));
plot(ZTCFQ.Time,ZTCFQ.TorsoTorque(:,3));
plot(ZTCFQ.Time,ZTCFQ.LScapTorque(:,3));
plot(ZTCFQ.Time,ZTCFQ.RScapTorque(:,3));
plot(ZTCFQ.Time,ZTCFQ.LSTorqueLocal(:,3));
plot(ZTCFQ.Time,ZTCFQ.RSTorqueLocal(:,3));
plot(ZTCFQ.Time,ZTCFQ.LeftElbowTorqueLocal(:,3));
plot(ZTCFQ.Time,ZTCFQ.RightElbowTorque(:,3));
plot(ZTCFQ.Time,ZTCFQ.LeftWristTorqueLocal(:,3));
plot(ZTCFQ.Time,ZTCFQ.RightWristTorqueLocal(:,3));

ylabel('Torque (Nm)');
grid 'on';

%Add Legend to Plot
legend('Hip Torque','Torso Torque', 'Left Scap Torque','Right Scap Torque','Left Shoulder Torque','Right Shoulder Torque','Left Elbow Torque','Right Elbow Torque','Left Wrist Torque','Right Wrist Torque');
legend('Location','southeast');

%Add a Title
title('Joint Torque Inputs');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Joint Torque Inputs');
pause(PauseTime);

%Close Figure
close(305);