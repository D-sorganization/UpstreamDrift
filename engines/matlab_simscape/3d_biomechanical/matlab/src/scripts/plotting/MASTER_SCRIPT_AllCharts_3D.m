function MASTER_SCRIPT_AllCharts_3D(BASEQ, ZTCFQ, DELTAQ, ZVCFQ, ModelData)
% MASTER_SCRIPT_AllCharts_3D - One-call runner for all consolidated quiver and plot charts
% Replaces legacy SCRIPT_AllPlots_3D.m by calling modern unified plot functions

%% Ensure output directories exist
mkdir('BaseData Charts');
mkdir('ZTCF Charts');
mkdir('Delta Charts');
mkdir('ZVCF Charts');
mkdir('Model Data Charts');
mkdir('Comparison Charts');
mkdir('BaseData Quiver Plots');
mkdir('ZTCF Quiver Plots');
mkdir('Delta Quiver Plots');
mkdir('ZVCF Quiver Plots');
mkdir('Model Data Quiver Plots');
mkdir('Comparison Quiver Plots');

%% Run master quiver plots
PLOT_BASEQuiver(BASEQ);
PLOT_ZTCFQuiver(ZTCFQ);
PLOT_DELTAQuiver(BASEQ, DELTAQ, ZTCFQ);
PLOT_ZVCFQuiver(ZVCFQ, DELTAQ, BASEQ);
PLOT_ModelDataQuiver(ModelData);
PLOT_ComparisonQuiver(BASEQ, DELTAQ);

%% Run master line/time-series plots
PLOT_BASE_Plots(BASEQ);
PLOT_ZTCF_Plots(ZTCFQ);
PLOT_DELTA_Plots(DELTAQ);
PLOT_ZVCF_Plots(ZVCFQ);
PLOT_ModelData_Plots(ModelData);
PLOT_Comparison_Plots(BASEQ, ZTCFQ, DELTAQ);

disp('All chart and quiver plots completed successfully.');
end
