figure(931);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LSonArmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LSonArmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLSPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Power','LS Angular Power','LS Total Power');
legend('Location','southeast');

%Add a Title
title('Left Shoulder Power on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Shoulder Power');
pause(PauseTime);

%Close Figure
close(931);