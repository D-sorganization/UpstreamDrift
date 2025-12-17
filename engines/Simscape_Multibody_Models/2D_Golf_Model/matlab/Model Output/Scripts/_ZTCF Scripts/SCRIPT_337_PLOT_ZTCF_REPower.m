figure(337);
hold on;

plot(ZTCFQ.Time,ZTCFQ.REonForearmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.REonForearmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalREPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RE Linear Power','RE Angular Power','RE Total Power');
legend('Location','southeast');

%Add a Title
title('Right Elbow Power on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Elbow Power');
pause(PauseTime);

%Close Figure
close(337);