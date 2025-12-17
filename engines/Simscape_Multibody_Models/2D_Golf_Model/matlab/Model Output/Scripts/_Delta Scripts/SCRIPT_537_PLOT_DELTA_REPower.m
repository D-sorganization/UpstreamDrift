figure(537);
hold on;

plot(DELTAQ.Time,DELTAQ.REonForearmLinearPower);
plot(DELTAQ.Time,DELTAQ.REonForearmAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalREPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RE Linear Power','RE Angular Power','RE Total Power');
legend('Location','southeast');

%Add a Title
title('Right Elbow Power on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Elbow Power');
pause(PauseTime);

%Close Figure
close(537);