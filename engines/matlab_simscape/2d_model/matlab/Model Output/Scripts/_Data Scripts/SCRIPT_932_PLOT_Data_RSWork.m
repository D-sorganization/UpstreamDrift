figure(932);
hold on;

plot(Data.Time,Data.RSLinearWorkonArm);
plot(Data.Time,Data.RSAngularWorkonArm);
plot(Data.Time,Data.TotalRSWork);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('RS Linear Work','RS Angular Work','RS Total Work');
legend('Location','southeast');

%Add a Title
title('Right Shoulder Work on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Right Shoulder Work');
pause(PauseTime);

%Close Figure
close(932);