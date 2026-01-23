figure(710);
hold on;

plot(BASEQ.Time,BASEQ.TotalLSPower);
plot(BASEQ.Time,BASEQ.TotalRSPower);
plot(BASEQ.Time,BASEQ.TotalLEPower);
plot(BASEQ.Time,BASEQ.TotalREPower);
plot(BASEQ.Time,BASEQ.TotalLWPower);
plot(BASEQ.Time,BASEQ.TotalRWPower);

plot(ZTCFQ.Time,ZTCFQ.TotalLSPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRSPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalLEPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalREPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalLWPower,'--');
plot(ZTCFQ.Time,ZTCFQ.TotalRWPower,'--');


ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Total Power - BASE','RS Total Power - BASE','LE Total Power - BASE','RE Total Power - BASE','LW Total Power - BASE','RW Total Power - BASE','LS Total Power - ZTCF','RS Total Power - ZTCF','LE Total Power - ZTCF','RE Total Power - ZTCF','LW Total Power - ZTCF','RW Total Power - ZTCF');
legend('Location','southeast');

%Add a Title
title('Total Power on Distal Segment');
subtitle('COMPARISON');

%Save Figure
savefig('Comparison Charts/COMPARISON_PLOT - Total Power on Distal');
pause(PauseTime);

%Close Figure
close(710);