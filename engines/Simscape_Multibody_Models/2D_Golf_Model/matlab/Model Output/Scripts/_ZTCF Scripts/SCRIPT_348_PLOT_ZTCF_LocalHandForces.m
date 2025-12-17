figure(348);
hold on;

plot(ZTCFQ.Time(:,1),ZTCFQ.LeftWristForceLocal(:,1));
plot(ZTCFQ.Time(:,1),ZTCFQ.LeftWristForceLocal(:,2));
plot(ZTCFQ.Time(:,1),ZTCFQ.RightWristForceLocal(:,1));
plot(ZTCFQ.Time(:,1),ZTCFQ.RightWristForceLocal(:,2));

ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('Left Wrist X','Left Wrist Y','Right Wrist X','Right Wrist Y');
legend('Location','southeast');

%Add a Title
title('Local Hand Forces on Club');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Local Hand Forces');
pause(PauseTime);

%Close Figure
close(348);