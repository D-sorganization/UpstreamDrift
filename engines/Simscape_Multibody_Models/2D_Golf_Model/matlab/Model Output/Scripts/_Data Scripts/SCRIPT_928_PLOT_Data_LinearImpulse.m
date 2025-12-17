figure(928);
hold on;

plot(Data.Time,Data.LHLinearImpulseonClub);
plot(Data.Time,Data.RHLinearImpulseonClub);
plot(Data.Time,Data.LinearImpulseonClub);

ylabel('Linear Impulse (kgm/s)');
grid 'on';

%Add Legend to Plot
legend('LH Linear Impulse','RH Linear Impulse','Net Force Linear Impulse (midpoint)');
legend('Location','southeast');

%Add a Title
title('Linear Impulse');
subtitle('Data');
%subtitle('Left Hand, Right Hand, Total');

%Save Figure
savefig('Data Charts/Plot - Linear Impulse LH,RH,Total');
pause(PauseTime);

%Close Figure
close(928);