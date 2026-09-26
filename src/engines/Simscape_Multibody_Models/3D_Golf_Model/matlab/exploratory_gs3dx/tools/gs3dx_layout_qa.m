function report = gs3dx_layout_qa(systems, out_dir)
%GS3DX_LAYOUT_QA  Snapshot GS3DX diagrams and list overlapping blocks (#10959).
%
%   REPORT = GS3DX_LAYOUT_QA(SYSTEMS, OUT_DIR) loads the model of each
%   system path in SYSTEMS (cellstr), prints the diagram to
%   OUT_DIR/<name>.png, and lists pairs of blocks whose rectangles overlap.
%   Nothing is saved.  REPORT is a struct array: .system, .png,
%   .n_blocks, .overlaps ({name1, name2} rows).

    arguments
        systems (1,:) cell
        out_dir (1,:) char
    end
    if ~isfolder(out_dir)
        mkdir(out_dir);
    end
    report = struct('system', {}, 'png', {}, 'n_blocks', {}, 'overlaps', {});
    for k = 1:numel(systems)
        sys = systems{k};
        load_system(strtok(sys, '/'));
        blocks = find_system(sys, 'SearchDepth', 1, 'LookUnderMasks', 'all', 'Type', 'Block');
        blocks = blocks(~strcmp(blocks, sys));
        rects = cell2mat(get_param(blocks, 'Position'));
        overlaps = cell(0, 2);
        for a = 1:numel(blocks)
            for b = a + 1:numel(blocks)
                if local_intersect(rects(a, :), rects(b, :))
                    overlaps(end + 1, :) = {get_param(blocks{a}, 'Name'), get_param(blocks{b}, 'Name')}; %#ok<AGROW>
                end
            end
        end
        png = fullfile(out_dir, [matlab.lang.makeValidName(strrep(sys, '/', '_')) '.png']);
        print(['-s' sys], '-dpng', png);
        report(end + 1) = struct('system', sys, 'png', png, 'n_blocks', numel(blocks), ...
            'overlaps', {overlaps}); %#ok<AGROW>
    end
end

function tf = local_intersect(r, s)
% Rectangles [left top right bottom] share interior area.
    tf = r(1) < s(3) && s(1) < r(3) && r(2) < s(4) && s(2) < r(4);
end
