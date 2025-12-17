figure(505);
hold on;

plot(DELTAQ.Time,DELTAQ.HipTorque(:,3));
plot(DELTAQ.Time,DELTAQ.TorsoTorque(:,3));
plot(DELTAQ.Time,DELTAQ.LScapTorque(:,3));
plot(DELTAQ.Time,DELTAQ.RScapTorque(:,3));
plot(DELTAQ.Time,DELTAQ.LSTorqueLocal(:,3));
plot(DELTAQ.Time,DELTAQ.RSTorqueLocal(:,3));
plot(DELTAQ.Time,DELTAQ.LeftElbowTorqueLocal(:,3));
plot(DELTAQ.Time,DELTAQ.RightElbowTorque(:,3));
plot(DELTAQ.Time,DELTAQ.LeftWristTorqueLocal(:,3));
plot(DELTAQ.Time,DELTAQ.RightWristTorqueLocal(:,3));

ylabel('Torque (Nm)');
grid 'on';

%Add Legend to Plot
legend('Hip Torque','Torso Torque', 'Left Scap Torque','Right Scap Torque','Left Shoulder Torque','Right Shoulder Torque','Left Elbow Torque','Right Elbow Torque','Left Wrist Torque','Right Wrist Torque');
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