% Need to write a script to generate a table of states from the xout data

% Current issue with this script is that some of the table variable names
% are empty as some of the data in the xout (states) data doesn't have a
% name.

%Generate a table with a time column and the variable name set to time.
Time=out.tout;
StatesData = table(Time,'VariableNames', {'Time'});

%Loop through each dataset element to add it to the table
for i=1:out.xout.numElements;
    %Get signal name
    signalName=out.xout.getElement(i).Name;
    %Get signal data
    signalData=out.xout.getElement(i).Values.Data;
    %Add the data as a new column in the table
    StatesData.(signalName)=signalData;
end

%Clean Up Workspace After Running and Generating the Table
clear i;
clear signalName;
clear signalData;
clear Time;