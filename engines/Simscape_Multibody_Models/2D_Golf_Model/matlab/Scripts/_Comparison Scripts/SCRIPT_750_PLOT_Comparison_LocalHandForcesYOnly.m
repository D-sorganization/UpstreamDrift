figure(750);
hold on;

plot(BASEQ.Time(:,1),BASEQ.LeftWristForceLocal(:,2),'LineWidth',3);
plot(BASEQ.Time(:,1),BASEQ.RightWristForceLocal(:,2),'LineWidth',3);

plot(ZTCFQ.Time(:,1),ZTCFQ.LeftWristForceLocal(:,2),'--','LineWidth',3);
plot(ZTCFQ.Time(:,1),ZTCFQ.RightWristForceLocal(:,2),'--','LineWidth',3);

ylabel('Force (N)');
grid 'on';

%Add Legend to Plot
legend('BASE Left Wrist Y','BASE Right Wrist Y','ZTCF Left Wrist Y','ZTCF Right Wrist Y');
legend('Location','southeast');

%Add a Title
title('Local Hand Forces on Club');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Local Hand Forces Y Only');
pause(PauseTime);

%Close Figure
close(750);