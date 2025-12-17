figure(135);
hold on;

plot(BASEQ.Time,BASEQ.LEonForearmLinearPower);
plot(BASEQ.Time,BASEQ.LEonForearmAngularPower);
plot(BASEQ.Time,BASEQ.TotalLEPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LE Linear Power','LE Angular Power','LE Total Power');
legend('Location','southeast');

%Add a Title
title('Left Elbow Power on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Elbow Power');
pause(PauseTime);

%Close Figure
close(135);