figure(136);
hold on;

plot(BASEQ.Time,BASEQ.RELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.REAngularWorkonForearm);
plot(BASEQ.Time,BASEQ.TotalREWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RE Linear Work','RE Angular Work','RE Total Work');
legend('Location','southeast');

%Add a Title
title('Right Elbow Work on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Elbow Work');
pause(PauseTime);

%Close Figure
close(136);