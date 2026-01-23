figure(104);
hold on;

plot(BASEQ.Time,BASEQ.LSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.RSLinearWorkonArm);
plot(BASEQ.Time,BASEQ.LELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.RELinearWorkonForearm);
plot(BASEQ.Time,BASEQ.LHLinearWorkonClub);
plot(BASEQ.Time,BASEQ.RHLinearWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Work','RS Linear Work','LE Linear Work','RE Linear Work','LW Linear Work','RW Linear Work');
legend('Location','southeast');

%Add a Title
title('Linear Work on Distal Segment');
subtitle('BASE');

%Save Figure
savefig('BaseData Charts/BASE_Plot - Linear Work on Distal');
pause(PauseTime);

%Close Figure
close(104);