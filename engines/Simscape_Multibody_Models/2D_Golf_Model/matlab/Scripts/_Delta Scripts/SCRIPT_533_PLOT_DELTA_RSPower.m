figure(533);
hold on;

plot(DELTAQ.Time,DELTAQ.RSonArmLinearPower);
plot(DELTAQ.Time,DELTAQ.RSonArmAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalRSPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RS Linear Power','RS Angular Power','RS Total Power');
legend('Location','southeast');

%Add a Title
title('Right Shoulder Power on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Shoulder Power');
pause(PauseTime);

%Close Figure
close(533);