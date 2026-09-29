figure(927);
hold on;

plot(Data.Time,Data.LHLinearWorkonClub);
plot(Data.Time,Data.RHLinearWorkonClub);
plot(Data.Time,Data.LinearWorkonClub);

ylabel('Work (J)');
grid 'on';

%Add Legend to Plot
legend('LH Linear Work','RH Linear Work','Net Force Linear Work (midpoint)');
legend('Location','southeast');

%Add a Title
title('Linear Work');
subtitle('Data');
%subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('Data Charts/Plot - Linear Work');
pause(PauseTime);

%Close Figure
close(927);