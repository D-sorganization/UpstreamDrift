figure(339);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LWonClubLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LWonClubAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LW Linear Power','LW Angular Power','LW Total Power');
legend('Location','southeast');

%Add a Title
title('Left Wrist Power on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Wrist Power');
pause(PauseTime);

%Close Figure
close(339);