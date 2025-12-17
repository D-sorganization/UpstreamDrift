figure(307);
hold on;

plot(ZTCFQ.Time,ZTCFQ.TotalLSPower);
plot(ZTCFQ.Time,ZTCFQ.TotalRSPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLEPower);
plot(ZTCFQ.Time,ZTCFQ.TotalREPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLWPower);
plot(ZTCFQ.Time,ZTCFQ.TotalRWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Total Power','RS Total Power','LE Total Power','RE Total Power','LW Total Power','RW Total Power');
legend('Location','southeast');

%Add a Title
title('Total Power on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Total Power');
pause(PauseTime);

%Close Figure
close(307);