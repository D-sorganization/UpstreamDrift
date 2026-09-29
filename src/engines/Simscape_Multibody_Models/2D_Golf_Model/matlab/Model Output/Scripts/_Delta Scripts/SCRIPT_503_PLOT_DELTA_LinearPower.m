figure(503);
hold on;

plot(ZTCFQ.Time,DELTAQ.LSonArmLinearPower);
plot(ZTCFQ.Time,DELTAQ.RSonArmLinearPower);
plot(ZTCFQ.Time,DELTAQ.LEonForearmLinearPower);
plot(ZTCFQ.Time,DELTAQ.REonForearmLinearPower);
plot(ZTCFQ.Time,DELTAQ.LWonClubLinearPower);
plot(ZTCFQ.Time,DELTAQ.RWonClubLinearPower);

ylabel('Power (W)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Power','RS Linear Power','LE Linear Power','RE Linear Power','LW Linear Power','RW Linear Power');
legend('Location','southeast');

%Add a Title
title('Linear Power on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Linear Power');
pause(PauseTime);

%Close Figure
close(503);