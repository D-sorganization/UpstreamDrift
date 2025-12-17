%Reads from a Table named Data in main workspace Generated from the Quivers
%Plots tab on the main worksheets. 

%Generate Shaft Quivers
Shaft=quiver3(Data.RWx(:,1),Data.RWy(:,1),Data.RWz(:,1),Data.Shaftdx(:,1),Data.Shaftdy(:,1),Data.Shaftdz(:,1),0);
Shaft.ShowArrowHead='off';		%Turn off arrow heads
Shaft.LineWidth=1;			    %Adjust line weighting
Shaft.Color=[0.5 0.5 0.5];		%Set shaft color gray
hold on;				        %Hold the current plot when you generate new

%Generate Grip Quivers
Grip=quiver3(Data.Buttx(:,1),Data.Butty(:,1),Data.Buttz(:,1),Data.Gripdx(:,1),Data.Gripdy(:,1),Data.Gripdz(:,1),0);
Grip.ShowArrowHead='off';
Grip.LineWidth=1;			    %Set grip line width
Grip.Color=[0 0 0];			    %Set grip color to black


