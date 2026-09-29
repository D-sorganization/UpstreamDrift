figure(541);
hold on;

plot(DELTAQ.Time,DELTAQ.RWonClubLinearPower);
plot(DELTAQ.Time,DELTAQ.RWonClubAngularPower);
plot(DELTAQ.Time,DELTAQ.TotalRWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RW Linear Power','RW Angular Power','RW Total Power');
legend('Location','southeast');

%Add a Title
title('Right Wrist Power on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Wrist Power');
pause(PauseTime);

%Close Figure
close(541);