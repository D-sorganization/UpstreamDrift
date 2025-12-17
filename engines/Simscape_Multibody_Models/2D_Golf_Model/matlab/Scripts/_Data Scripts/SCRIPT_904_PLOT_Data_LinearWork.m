figure(904);
hold on;

plot(Data.Time,Data.LSLinearWorkonArm);
plot(Data.Time,Data.RSLinearWorkonArm);
plot(Data.Time,Data.LELinearWorkonForearm);
plot(Data.Time,Data.RELinearWorkonForearm);
plot(Data.Time,Data.LHLinearWorkonClub);
plot(Data.Time,Data.RHLinearWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Linear Work','RS Linear Work','LE Linear Work','RE Linear Work','LW Linear Work','RW Linear Work');
legend('Location','southeast');

%Add a Title
title('Linear Work on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Linear Work on Distal');
pause(PauseTime);

%Close Figure
close(904);