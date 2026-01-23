figure(130);
hold on;

plot(BASEQ.Time,BASEQ.LSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.LSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.TotalLSWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Work','LS Angular Work','LS Total Work');
legend('Location','southeast');

%Add a Title
title('Left Shoulder Work on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Left Shoulder Work');
pause(PauseTime);

%Close Figure
close(130);