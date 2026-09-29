figure(333);
hold on;

plot(ZTCFQ.Time,ZTCFQ.RSonArmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.RSonArmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalRSPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RS Linear Power','RS Angular Power','RS Total Power');
legend('Location','southeast');

%Add a Title
title('Right Shoulder Power on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Shoulder Power');
pause(PauseTime);

%Close Figure
close(333);