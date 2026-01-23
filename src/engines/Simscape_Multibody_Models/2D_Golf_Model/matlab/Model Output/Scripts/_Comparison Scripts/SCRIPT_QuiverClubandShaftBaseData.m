%Reads from a Table named Data in main workspace Generated from the Quivers
%Plots tab on the main worksheets. 

%Generate Shaft Quivers
Shaft=quiver3(BASEQ.RWx(:,1),BASEQ.RWy(:,1),BASEQ.RWz(:,1),BASEQ.Shaftdx(:,1),BASEQ.Shaftdy(:,1),BASEQ.Shaftdz(:,1),0);
Shaft.ShowArrowHead='off';		%Turn off arrow heads
Shaft.LineWidth=1;			    %Adjust line weighting
Shaft.Color=[0.5 0.5 0.5];		%Set shaft color gray
hold on;				        %Hold the current plot when you generate new

%Generate Grip Quivers
Grip=quiver3(BASEQ.Buttx(:,1),BASEQ.Butty(:,1),BASEQ.Buttz(:,1),BASEQ.Gripdx(:,1),BASEQ.Gripdy(:,1),BASEQ.Gripdz(:,1),0);
Grip.ShowArrowHead='off';
Grip.LineWidth=1;			    %Set grip line width
Grip.Color=[0 0 0];			    %Set grip color to black

%Calculate height of table
H=height(BASEQ);

%Calculate how many rows to copy
h=H-1;

%Generate Hand Path Quivers
HandPath=quiver3(BASEQ.MPx(1:h,1),BASEQ.MPy(1:h,1),BASEQ.MPz(1:h,1),BASEQ.MPPx(1:h,1),BASEQ.MPPy(1:h,1),BASEQ.MPPz(1:h,1),0);
HandPath.ShowArrowHead='off';		%Turn off arrow heads
HandPath.LineWidth=1;			    %Adjust line weighting
HandPath.Color=[0 0 0];		        %Set shaft color black

%Generate Club Path Quivers
ClubPath=quiver3(BASEQ.CHx(1:h,1),BASEQ.CHy(1:h,1),BASEQ.CHz(1:h,1),BASEQ.CHPx(1:h,1),BASEQ.CHPy(1:h,1),BASEQ.CHPz(1:h,1),0);
ClubPath.ShowArrowHead='off';		%Turn off arrow heads
ClubPath.LineWidth=1;			    %Adjust line weighting
ClubPath.Color=[0 0 0];		        %Set shaft color black

% clear Grip;
% clear Shaft;
% clear HandPath;
% clear ClubPath;
clear h;
clear H;