figure(531);
hold on;

plot(DELTAQ.Time,DELTAQ.LSonArmLinearPower);
plot(DELTAQ.Time,DELTAQ.LSonArmAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalLSPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Power','LS Angular Power','LS Total Power');
legend('Location','southeast');

%Add a Title
title('Left Shoulder Power on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Shoulder Power');
pause(PauseTime);

%Close Figure
close(531);