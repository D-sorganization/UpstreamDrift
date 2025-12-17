figure(133);
hold on;

plot(BASEQ.Time,BASEQ.RSonArmLinearPower);
plot(BASEQ.Time,BASEQ.RSonArmAngularPower);
plot(BASEQ.Time,BASEQ.TotalRSPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('RS Linear Power','RS Angular Power','RS Total Power');
legend('Location','southeast');

%Add a Title
title('Right Shoulder Power on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Shoulder Power');
pause(PauseTime);

%Close Figure
close(133);