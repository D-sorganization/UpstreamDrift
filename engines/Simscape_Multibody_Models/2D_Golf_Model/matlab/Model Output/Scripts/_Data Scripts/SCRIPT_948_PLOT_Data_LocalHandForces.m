figure(948);
hold on;

plot(Data.Time(:,1),Data.LeftWristForceLocal(:,1));
plot(Data.Time(:,1),Data.LeftWristForceLocal(:,2));
plot(Data.Time(:,1),Data.RightWristForceLocal(:,1));
plot(Data.Time(:,1),Data.RightWristForceLocal(:,2));

ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('Left Wrist X','Left Wrist Y','Right Wrist X','Right Wrist Y');
legend('Location','southeast');

%Add a Title
title('Local Hand Forces on Club');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Local Hand Forces');
pause(PauseTime);

%Close Figure
close(948);