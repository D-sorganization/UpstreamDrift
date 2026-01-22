figure(535);
hold on;

plot(DELTAQ.Time,DELTAQ.LEonForearmLinearPower);
plot(DELTAQ.Time,DELTAQ.LEonForearmAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalLEPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LE Linear Power','LE Angular Power','LE Total Power');
legend('Location','southeast');

%Add a Title
title('Left Elbow Power on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Elbow Power');
pause(PauseTime);

%Close Figure
close(535);