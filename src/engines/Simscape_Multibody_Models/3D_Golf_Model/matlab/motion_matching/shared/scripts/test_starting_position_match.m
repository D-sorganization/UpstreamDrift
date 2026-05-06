function result = test_starting_position_match(varargin)
%TEST_STARTING_POSITION_MATCH  End-to-end check of model-vs-measured starting pose.
%
%   RESULT = TEST_STARTING_POSITION_MATCH() loads the 3D Simscape model
%   with the Impact input file, loads the measured Wiffle ProV1 club path,
%   and opens the interactive offset-tuning figure (PLOT_STARTING_POSITION_MATCH).
%
%   RESULT = TEST_STARTING_POSITION_MATCH('mode','batch', ...) runs the
%   same pipeline non-interactively, saving a screenshot and returning a
%   struct with the t=0 RMSE numbers so this can be invoked from
%   `matlab -batch`.
%
%   Name/value arguments:
%     'mode'         'gui'  | 'batch'      (default 'gui')
%     'sheet'        Excel sheet name      (default 'TW_ProV1')
%     'xlsx_path'    override for the Wiffle file location
%     'output_dir'   where to save batch artifacts
%
%   Output (struct):
%     .skel              full skeleton struct from load_impact_starting_position
%     .target            target struct from load_club_target_excel
%     .butt_error_mm     mm distance, model butt   ↔ measured butt    (no offset)
%     .clubhead_error_mm mm distance, model CH     ↔ measured CH      (no offset)
%     .png_path          path of saved figure (batch mode)
%
%   See also: LOAD_IMPACT_STARTING_POSITION, PLOT_STARTING_POSITION_MATCH,
%             LOAD_CLUB_TARGET_EXCEL.

    p = inputParser();
    addParameter(p, 'mode', 'gui', @(s) any(strcmpi(s, {'gui','batch'})));
    addParameter(p, 'sheet', 'TW_ProV1', @(s) ischar(s) || isstring(s));
    addParameter(p, 'xlsx_path', '', @(s) ischar(s) || isstring(s));
    addParameter(p, 'output_dir', '', @(s) ischar(s) || isstring(s));
    parse(p, varargin{:});
    args = p.Results;

    % --- Resolve project paths and add helpers to the path. -----------
    here     = fileparts(mfilename('fullpath'));
    shared   = fileparts(here);                    % motion_matching/shared
    mm_root  = fileparts(shared);                  % motion_matching
    mat_root = fileparts(mm_root);                 % matlab/
    addpath(shared);
    addpath(genpath(fullfile(mat_root, 'src')));

    % --- Resolve measured Excel file. ---------------------------------
    if isempty(char(args.xlsx_path))
        args.xlsx_path = fullfile(mat_root, 'src', 'apps', 'golf_gui', ...
            'Motion Capture Plotter', 'Wiffle_ProV1_club_3D_data.xlsx');
    end
    if ~isfile(args.xlsx_path)
        error('test_starting_position_match:noExcel', ...
              'Measured Wiffle Excel not found at %s', args.xlsx_path);
    end

    % --- Load the model's t=0 skeleton. ------------------------------
    fprintf('=== Step 1: load Impact starting position from model ===\n');
    skel = load_impact_starting_position(struct('verbose', true));

    % --- Load the measured club path. --------------------------------
    fprintf('\n=== Step 2: load measured Wiffle ProV1 club path ===\n');
    target = load_club_target_excel(string(args.xlsx_path), string(args.sheet));

    % --- Compute zero-offset error (model_butt vs measured_butt at impact). ---
    idx = double(target.impact_idx);
    butt_err_mm = 1000 * norm(skel.butt - target.butt(idx, :));
    ch_err_mm   = 1000 * norm(skel.ch   - target.clubhead(idx, :));
    fprintf('\n=== Step 3: zero-offset error (model frame == measured frame assumption) ===\n');
    fprintf('   butt:     %7.1f mm\n', butt_err_mm);
    fprintf('   clubhead: %7.1f mm  (impact frame %d, t=%.3fs)\n', ...
            ch_err_mm, idx, target.time(idx));

    result = struct( ...
        'skel', skel, ...
        'target', target, ...
        'butt_error_mm', butt_err_mm, ...
        'clubhead_error_mm', ch_err_mm, ...
        'png_path', "");

    % --- GUI vs batch dispatch. --------------------------------------
    if strcmpi(args.mode, 'batch')
        if isempty(char(args.output_dir))
            args.output_dir = fullfile(mat_root, 'output', 'starting_position_match');
        end
        if ~exist(args.output_dir, 'dir'); mkdir(args.output_dir); end
        png = fullfile(args.output_dir, sprintf('starting_position_%s.png', ...
                                                 datestr(now, 'yyyymmdd_HHMMSS'))); %#ok<TNOW1,DATST>
        fig = plot_starting_position_match(skel, target, ...
                struct('visible', 'off', 'save_png', png));
        close(fig);
        result.png_path = string(png);
        fprintf('\n=== Step 4 (batch): saved figure to %s ===\n', png);
    else
        fprintf('\n=== Step 4 (gui): launching interactive comparison ===\n');
        plot_starting_position_match(skel, target);
    end
end
