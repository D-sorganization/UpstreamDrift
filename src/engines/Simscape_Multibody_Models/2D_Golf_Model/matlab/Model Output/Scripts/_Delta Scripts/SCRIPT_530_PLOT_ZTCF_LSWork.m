figure(530);
hold on;

plot(DELTAQ.Time,DELTAQ.LSLinearWorkonArm);
plot(DELTAQ.Time,DELTAQ.LSAngularWorkonArm);
plot(DELTAQ.Time,DELTAQ.TotalLSWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Work','LS Angular Work','LS Total Work');
legend('Location','southeast');

%Add a Title
title('Left Shoulder Work on Distal Segment');
subtitle('DELTA');

%Save Figure
savefig('Delta Charts/DELTA_Plot - Left Shoulder Work');
pause(PauseTime);

%Close Figure
close(530);