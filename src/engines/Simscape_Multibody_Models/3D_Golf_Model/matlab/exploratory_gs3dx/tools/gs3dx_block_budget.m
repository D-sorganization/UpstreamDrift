function report = gs3dx_block_budget(mdl, opts)
%GS3DX_BLOCK_BUDGET  Report block-budget breakdown and Simscape library metrics.
%
%   REPORT = GS3DX_BLOCK_BUDGET(MDL) loads the model MDL (if not already
%   loaded) and returns:
%     .model                Model name as string
%     .nonvirtual_total     find_system(mdl,'LookUnderMasks','all','FollowLinks','on',...
%                             'MatchFilter',@Simulink.match.activeVariants,...
%                             'Type','Block','Virtual','off')
%     .by_type              Table of BlockType plus ReferenceBlock (newlines
%                           replaced by spaces) with counts, sorted descending
%     .converter_internal   Count of nonvirtual blocks inside nesl_utility
%                           PS-Simulink/Simulink-PS converters
%     .top_level_equivalent Nonvirtual count when each converter is counted as 1 block
%     .simscape_blocks      Count of nonvirtual blocks whose ReferenceBlock starts
%                           with sm_lib, fl_lib, nesl_utility, or ee_lib
%
%   REPORT = GS3DX_BLOCK_BUDGET(MDL, json_path=PATH) additionally writes
%   the report to PATH as formatted JSON.
%
%   REPORT = GS3DX_BLOCK_BUDGET(MDL, compiled=true) also compiles MDL
%   (update diagram, nothing saved) and adds
%     .compiled_total       the same count after compilation.
%   That is the number the Home license limits: compiling adds Simscape's
%   own blocks (GS3DX_Quat 594 -> 740, GS3DX_FullBody 751 -> 945).  A model
%   over the limit fails to compile, which raises the license error here.
%
%   Preconditions:
%     - MDL is a non-empty text scalar resolving to a Simulink model.
%   Postconditions:
%     - nonvirtual_total >= 0
%     - 0 <= converter_internal <= nonvirtual_total
%     - top_level_equivalent >= 0
%     - 0 <= simscape_blocks <= nonvirtual_total
%     - istable(by_type)

    arguments
        mdl {mustBeTextScalar}
        opts.json_path {mustBeTextScalar} = ''
        opts.compiled (1,1) logical = false
    end

    mdl_str = char(mdl);
    assert(~isempty(strtrim(mdl_str)), 'gs3dx:invalidModel', 'Model name cannot be empty.');

    [~, stem] = fileparts(mdl_str);
    if isempty(stem)
        stem = mdl_str;
    end

    was_loaded = bdIsLoaded(stem);
    if ~was_loaded
        load_system(mdl_str);
        cleanup_bd = onCleanup(@() close_system(stem, 0));
    end

    % 1. Nonvirtual blocks
    nv = find_system(stem, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'MatchFilter', @Simulink.match.activeVariants, 'Type', 'Block', 'Virtual', 'off');
    nonvirtual_total = numel(nv);

    % 2. Table of BlockType plus ReferenceBlock (newlines replaced by spaces)
    if isempty(nv)
        by_type = table(string.empty(0, 1), string.empty(0, 1), zeros(0, 1), ...
            'VariableNames', {'BlockType', 'ReferenceBlock', 'Count'});
        rb_clean = string.empty(0, 1);
    else
        bt_clean = replace(string(get_param(nv, 'BlockType')), newline, ' ');
        rb_clean = replace(string(get_param(nv, 'ReferenceBlock')), newline, ' ');
        [G, u_bt, u_rb] = findgroups(bt_clean, rb_clean);
        cnt = splitapply(@numel, nv, G);
        by_type = table(u_bt, u_rb, cnt, 'VariableNames', {'BlockType', 'ReferenceBlock', 'Count'});
        by_type = sortrows(by_type, 'Count', 'descend');
    end

    % 3. Converter detection & internal nonvirtual block count
    conv_blocks = find_system(stem, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'MatchFilter', @Simulink.match.activeVariants, 'Type', 'Block');
    conv_refs = replace(string(get_param(conv_blocks, 'ReferenceBlock')), newline, ' ');
    is_conv_block = (conv_refs == "nesl_utility/PS-Simulink Converter" | ...
                     conv_refs == "nesl_utility/Simulink-PS Converter");
    num_converters = sum(is_conv_block);
    conv_list = conv_blocks(is_conv_block);

    if isempty(nv)
        converter_internal = 0;
    else
        is_conv_internal = startsWith(rb_clean, "nesl_utility/PS-Simulink Converter/") | ...
                           startsWith(rb_clean, "nesl_utility/Simulink-PS Converter/");
        if ~isempty(conv_list)
            for k = 1:numel(nv)
                if ~is_conv_internal(k)
                    p = get_param(nv{k}, 'Parent');
                    while ~isempty(p) && ~strcmp(p, stem)
                        if ismember(p, conv_list)
                            is_conv_internal(k) = true;
                            break;
                        end
                        p = get_param(p, 'Parent');
                    end
                end
            end
        end
        converter_internal = sum(is_conv_internal);
    end

    % 4. Top-level equivalent nonvirtual count
    top_level_equivalent = (nonvirtual_total - converter_internal) + num_converters;

    % 5. Simscape library blocks (sm_lib, fl_lib, nesl_utility, ee_lib)
    names = gs3dx_names();
    if isfield(names, 'simscape_prefixes')
        prefixes = names.simscape_prefixes;
    else
        prefixes = ["sm_lib", "fl_lib", "nesl_utility", "ee_lib"];
    end

    if isempty(nv)
        simscape_blocks = 0;
    else
        is_ss = false(size(nv));
        for pfx = prefixes
            is_ss = is_ss | startsWith(rb_clean, pfx);
        end
        simscape_blocks = sum(is_ss);
    end

    % Build report
    report = struct();
    report.model                = string(stem);
    report.nonvirtual_total     = nonvirtual_total;
    report.by_type              = by_type;
    report.converter_internal   = converter_internal;
    report.top_level_equivalent = top_level_equivalent;
    report.simscape_blocks      = simscape_blocks;
    if opts.compiled
        set_param(stem, 'SimulationCommand', 'update');
        report.compiled_total = numel(find_system(stem, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
            'MatchFilter', @Simulink.match.activeVariants, 'Type', 'Block', 'Virtual', 'off'));
        assert(report.compiled_total >= report.nonvirtual_total, 'gs3dx:postcondition', ...
            'Postcondition: compilation cannot remove nonvirtual blocks.');
    end

    % Postconditions
    assert(report.nonvirtual_total >= 0, 'gs3dx:postcondition', ...
        'Postcondition: nonvirtual_total must be non-negative.');
    assert(report.converter_internal >= 0 && report.converter_internal <= report.nonvirtual_total, ...
        'gs3dx:postcondition', 'Postcondition: converter_internal must be in [0, nonvirtual_total].');
    assert(report.top_level_equivalent >= 0, 'gs3dx:postcondition', ...
        'Postcondition: top_level_equivalent must be non-negative.');
    assert(report.simscape_blocks >= 0 && report.simscape_blocks <= report.nonvirtual_total, ...
        'gs3dx:postcondition', 'Postcondition: simscape_blocks must be in [0, nonvirtual_total].');
    assert(istable(report.by_type), 'gs3dx:postcondition', ...
        'Postcondition: by_type must be a table.');

    % JSON export
    json_path = char(opts.json_path);
    if ~isempty(json_path)
        out_dir = fileparts(json_path);
        if ~isempty(out_dir) && ~isfolder(out_dir)
            mkdir(out_dir);
        end
        fid = fopen(json_path, 'w');
        if fid == -1
            error('gs3dx:fileError', 'Cannot open file for writing: %s', json_path);
        end
        c_fid = onCleanup(@() fclose(fid));
        json_text = jsonencode(report, 'PrettyPrint', true);
        fprintf(fid, '%s\n', json_text);
    end
end
