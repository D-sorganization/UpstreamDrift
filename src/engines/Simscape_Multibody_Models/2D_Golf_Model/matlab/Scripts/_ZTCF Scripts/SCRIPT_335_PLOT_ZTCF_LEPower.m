figure(335);
hold on;

plot(ZTCFQ.Time,ZTCFQ.LEonForearmLinearPower);
plot(ZTCFQ.Time,ZTCFQ.LEonForearmAngularPower);
plot(ZTCFQ.Time,ZTCFQ.TotalLEPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LE Linear Power','LE Angular Power','LE Total Power');
legend('Location','southeast');

%Add a Title
title('Left Elbow Power on Distal Segment');
subtitle('ZTCF');

%Save Figure
savefig('ZTCF Charts/ZTCF_Plot - Left Elbow Power');
pause(PauseTime);

%Close Figure
close(335);