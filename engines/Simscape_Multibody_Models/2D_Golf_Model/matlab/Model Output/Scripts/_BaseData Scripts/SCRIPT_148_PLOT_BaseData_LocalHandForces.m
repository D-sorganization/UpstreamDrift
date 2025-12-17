figure(148);
hold on;

plot(BASEQ.Time(:,1),BASEQ.LeftWristForceLocal(:,1));
plot(BASEQ.Time(:,1),BASEQ.LeftWristForceLocal(:,2));
plot(BASEQ.Time(:,1),BASEQ.RightWristForceLocal(:,1));
plot(BASEQ.Time(:,1),BASEQ.RightWristForceLocal(:,2));

ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('Left Wrist X','Left Wrist Y','Right Wrist X','Right Wrist Y');
legend('Location','southeast');

%Add a Title
title('Local Hand Forces on Club');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Local Hand Forces');
pause(PauseTime);

%Close Figure
close(148);