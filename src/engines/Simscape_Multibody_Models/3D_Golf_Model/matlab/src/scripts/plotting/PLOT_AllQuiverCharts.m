function PLOT_AllQuiverCharts(BASEQ, ZTCFQ, DELTAQ, ZVCFQ, ModelData)
%PLOT_AllQuiverCharts - Master runner for all unified quiver plotting scripts
% This script sequentially calls all PLOT_*Quiver functions, saving outputs into their respective folders.

% Ensure output directories exist
mkdir('BaseData Quiver Plots');
mkdir('ZTCF Quiver Plots');
mkdir('Delta Quiver Plots');
mkdir('ZVCF Quiver Plots');
mkdir('Model Data Quiver Plots');
mkdir('Comparison Quiver Plots');

% Run all quiver suites (plot functions)
PLOT_BASEQuiver(BASEQ);
PLOT_ZTCFQuiver(ZTCFQ);
PLOT_DELTAQuiver(BASEQ, DELTAQ, ZTCFQ);
PLOT_ZVCFQuiver(ZVCFQ, DELTAQ, BASEQ);
PLOT_ModelDataQuiver(ModelData);
PLOT_ComparisonQuiver(BASEQ, DELTAQ);

disp('All quiver plots generated and saved.');
end
