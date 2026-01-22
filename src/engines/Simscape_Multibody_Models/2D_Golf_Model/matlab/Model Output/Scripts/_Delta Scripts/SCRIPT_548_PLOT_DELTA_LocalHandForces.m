figure(548);
hold on;

plot(DELTAQ.Time(:,1),DELTAQ.LeftWristForceLocal(:,1));
plot(DELTAQ.Time(:,1),DELTAQ.LeftWristForceLocal(:,2));
plot(DELTAQ.Time(:,1),DELTAQ.RightWristForceLocal(:,1));
plot(DELTAQ.Time(:,1),DELTAQ.RightWristForceLocal(:,2));

ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('Left Wrist X','Left Wrist Y','Right Wrist X','Right Wrist Y');
legend('Location','southeast');

%Add a Title
title('Local Hand Forces on Club');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Local Hand Forces');
pause(PauseTime);

%Close Figure
close(548);