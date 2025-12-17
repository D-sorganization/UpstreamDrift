figure(901);
hold on;

plot(Data.Time,Data.LSAngularWorkonArm);
plot(Data.Time,Data.RSAngularWorkonArm);
plot(Data.Time,Data.LEAngularWorkonForearm);
plot(Data.Time,Data.REAngularWorkonForearm);
plot(Data.Time,Data.LWAngularWorkonClub);
plot(Data.Time,Data.RWAngularWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LS Angular Work','RS Angular Work','LE Angular Work','RE Angular Work','LW Angular Work','RW Angular Work');
legend('Location','southeast');

%Add a Title
title('Angular Work on Distal Segment');
subtitle('Data');

%Save Figure
savefig('Data Charts/Data_Plot - Angular Work');
pause(PauseTime);

%Close Figure
close(901);