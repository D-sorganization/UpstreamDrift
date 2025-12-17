figure(532);
hold on;

plot(DELTAQ.Time,DELTAQ.RSLinearWorkonArm);
plot(DELTAQ.Time,DELTAQ.RSAngularWorkonArm);
plot(DELTAQ.Time,DELTAQ.TotalRSWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RS Linear Work','RS Angular Work','RS Total Work');
legend('Location','southeast');

%Add a Title
title('Right Shoulder Work on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Right Shoulder Work');
pause(PauseTime);

%Close Figure
close(532);