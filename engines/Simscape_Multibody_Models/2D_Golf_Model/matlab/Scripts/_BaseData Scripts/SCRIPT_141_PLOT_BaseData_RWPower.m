figure(141);
hold on;

plot(BASEQ.Time,BASEQ.RWonClubLinearPower);
plot(BASEQ.Time,BASEQ.RWonClubAngularPower);
plot(BASEQ.Time,BASEQ.TotalRWPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RW Linear Power','RW Angular Power','RW Total Power');
legend('Location','southeast');

%Add a Title
title('Right Wrist Power on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Wrist Power');
pause(PauseTime);

%Close Figure
close(141);