figure(748);
hold on;


plot(BASEQ.Time(:,1),BASEQ.LeftWristForceLocal(:,1),'LineWidth',3);
plot(BASEQ.Time(:,1),BASEQ.LeftWristForceLocal(:,2),'LineWidth',3);
plot(BASEQ.Time(:,1),BASEQ.RightWristForceLocal(:,1),'LineWidth',3);
plot(BASEQ.Time(:,1),BASEQ.RightWristForceLocal(:,2),'LineWidth',3);

plot(ZTCFQ.Time(:,1),ZTCFQ.LeftWristForceLocal(:,1),'--','LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.LeftWristForceLocal(:,2),'--','LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.RightWristForceLocal(:,1),'--','LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.RightWristForceLocal(:,2),'--','LineWidth',3);

ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('BASE Left Wrist X','BASE Left Wrist Y','BASE Right Wrist X','BASE Right Wrist Y','ZTCF Left Wrist X','ZTCF Left Wrist Y','ZTCF Right Wrist X','ZTCF Right Wrist Y');
legend('Location','southeast');

%Add a Title
title('Local Hand Forces on Club');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Local Hand Forces');
pause(PauseTime);

%Close Figure
close(748);