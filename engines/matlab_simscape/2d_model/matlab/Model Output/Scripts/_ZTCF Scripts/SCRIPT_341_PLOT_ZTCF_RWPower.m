figure(341);
hold on;

plot(ZTCFQ.Time,ZTCFQ.RWonClubLinearPower);
plot(ZTCFQ.Time,ZTCFQ.RWonClubAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalRWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RW Linear Power','RW Angular Power','RW Total Power');
legend('Location','southeast');

%Add a Title
title('Right Wrist Power on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Right Wrist Power');
pause(PauseTime);

%Close Figure
close(341);