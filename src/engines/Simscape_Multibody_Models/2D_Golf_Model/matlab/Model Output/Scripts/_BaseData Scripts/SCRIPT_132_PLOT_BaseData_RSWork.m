figure(132);
hold on;

plot(BASEQ.Time,BASEQ.RSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.RSAngularWorkonArm);
plot(BASEQ.Time,BASEQ.TotalRSWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RS Linear Work','RS Angular Work','RS Total Work');
legend('Location','southeast');

%Add a Title
title('Right Shoulder Work on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Right Shoulder Work');
pause(PauseTime);

%Close Figure
close(132);