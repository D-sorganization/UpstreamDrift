%Add Total Work and Power to Tables

%Generate Total Work and Power Vectors

%Data
Data.TotalLSWork=Data.LSAngularWorkonArm+Data.LSLinearWorkonArm;
Data.TotalRSWork=Data.RSAngularWorkonArm+Data.RSLinearWorkonArm;
Data.TotalLEWork=Data.LEAngularWorkonForearm+Data.LELinearWorkonForearm;
Data.TotalREWork=Data.REAngularWorkonForearm+Data.RELinearWorkonForearm;
Data.TotalLWWork=Data.LWAngularWorkonClub+Data.LHLinearWorkonClub;
Data.TotalRWWork=Data.RWAngularWorkonClub+Data.RHLinearWorkonClub;
Data.TotalLSPower=Data.LSonArmAngularPower+Data.LSonArmLinearPower;
Data.TotalRSPower=Data.RSonArmAngularPower+Data.RSonArmLinearPower;
Data.TotalLEPower=Data.LEonForearmAngularPower+Data.LEonForearmLinearPower;
Data.TotalREPower=Data.REonForearmAngularPower+Data.REonForearmLinearPower;
Data.TotalLWPower=Data.LWonClubAngularPower+Data.LWonClubLinearPower;
Data.TotalRWPower=Data.RWonClubAngularPower+Data.RWonClubLinearPower;

