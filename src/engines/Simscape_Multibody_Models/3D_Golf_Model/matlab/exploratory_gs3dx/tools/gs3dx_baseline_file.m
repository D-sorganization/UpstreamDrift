function file = gs3dx_baseline_file(info, drive, stop_time)
%GS3DX_BASELINE_FILE  Path of the recorded original run for a drive.
%
%   FILE = GS3DX_BASELINE_FILE(INFO, DRIVE, STOP_TIME) returns
%   INFO.baselines_dir/original_GolfSwing3D_Kinetic_<drive>_<t>S.mat, e.g.
%   original_GolfSwing3D_Kinetic_impact_0p3S.mat.  See GS3DX_DRIVE.

    arguments
        info (1,1) struct
        drive (1,1) string
        stop_time (1,1) double {mustBePositive} = 0.3
    end
    names = gs3dx_names();
    file = fullfile(info.baselines_dir, sprintf('original_%s_%s_%sS.mat', ...
        names.original_model, drive, strrep(num2str(stop_time), '.', 'p')));
end
