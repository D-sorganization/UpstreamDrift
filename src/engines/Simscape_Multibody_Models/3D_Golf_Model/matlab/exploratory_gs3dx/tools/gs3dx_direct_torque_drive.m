function removed = gs3dx_direct_torque_drive(mdl)
%GS3DX_DIRECT_TORQUE_DRIVE  Drive joint torque axes straight from their converters.
%
%   REMOVED = GS3DX_DIRECT_TORQUE_DRIVE(MDL) rewires every torque axis of the
%   loaded subsystem model MDL that is driven through the 1-D chain
%
%       Simulink-PS Converter -> Ideal Torque Source -> Rotational Multibody
%       Interface -> joint InputTorque port   (sources grounded by Mechanical
%       Rotational References; each interface also reads a joint sense port)
%
%   so that the converter feeds the joint's InputTorque port directly, and
%   deletes the 1-D blocks.  Nets the 1-D blocks shared with other blocks
%   (joint sense ports that also feed converters) are redrawn between the
%   remaining ports.  The network is massless, so the torque reaching the
%   joint is unchanged; each axis saves three non-virtual blocks.  Returns
%   the number of axes rewired.  The caller saves.
%
%   Physical nets are found as connected components of all line segments:
%   a branched net lists its segments only on the trunk (LineChildren), and
%   branch segments report no parent or end ports, so walking from a port's
%   own segment is not enough.
%
%   Preconditions (checked before anything is edited):
%     - equal numbers of interfaces, sources and references;
%     - each interface's first port drives one joint port, each interface is
%       joined point-to-point to one source, and each source is fed
%       point-to-point by one Simulink-PS converter.
%   Postconditions:
%     - MDL contains none of the 1-D blocks;
%     - every other port that shared a net with a 1-D block is connected to
%       exactly the same other ports as before, plus its new converter/joint
%       partner, so no sensor or converter is disconnected.

    arguments
        mdl (1,:) char
    end
    lib.interface = 'fl_lib/Mechanical/Multibody Interfaces/Rotational Multibody Interface';
    lib.source    = 'fl_lib/Mechanical/Mechanical Sources/Ideal Torque Source';
    lib.reference = 'fl_lib/Mechanical/Rotational Elements/Mechanical Rotational Reference';
    lib.converter = 'nesl_utility/Simulink-PS Converter';
    find_ref = @(ref) find_system(mdl, 'SearchDepth', 1, 'ReferenceBlock', ref);
    handles  = @(c) reshape(cellfun(@(b) get_param(b, 'Handle'), c), 1, []);

    interfaces = find_ref(lib.interface);
    sources    = find_ref(lib.source);
    references = find_ref(lib.reference);
    assert(numel(sources) == numel(interfaces) && numel(references) == numel(sources), ...
        'gs3dx:directDrive', 'Precondition: %d interfaces, %d sources, %d references in %s', ...
        numel(interfaces), numel(sources), numel(references), mdl);
    one_d = handles([interfaces; sources; references]);
    g = local_graph(mdl);

    % Plan: (converter port, joint port) per axis.
    pairs = zeros(numel(interfaces), 2);
    for k = 1:numel(interfaces)
        ph = get_param(interfaces{k}, 'PortHandles');
        joint_port = local_partners(g, ph.LConn(1), one_d);
        src_port   = local_point_to_point(g, interfaces{k}, handles(sources));
        assert(isscalar(joint_port) && isscalar(src_port), 'gs3dx:directDrive', ...
            'Precondition: %s must drive one joint port and meet one torque source', interfaces{k});
        source    = get_param(src_port, 'Parent');
        conv_port = local_point_to_point(g, source, setdiff(g.owner, one_d));
        assert(isscalar(conv_port) && ...
            strcmp(local_ref(get_param(conv_port, 'Parent')), lib.converter), ...
            'gs3dx:directDrive', 'Precondition: %s must be fed by one Simulink-PS converter', source);
        pairs(k, :) = [conv_port, joint_port];
    end

    % Expected connectivity of every kept port on a touched net.
    nets = unique(g.net(ismember(g.owner, one_d) & g.net > 0));
    expected = containers.Map('KeyType', 'double', 'ValueType', 'any');
    reconnect = {};
    for n = nets
        kept = g.port(g.net == n & ~ismember(g.owner, one_d));
        for q = kept
            expected(q) = setdiff(kept, q);
        end
        if numel(kept) > 1
            reconnect{end+1} = kept; %#ok<AGROW>
        end
    end
    for k = 1:size(pairs, 1)
        for side = [1 2; 2 1].'
            q = pairs(k, side(1));
            expected(q) = union(expected(q), pairs(k, side(2)));
        end
    end

    % Delete every touched net and the 1-D blocks, then redraw.
    for x = g.line(ismember(g.line_net, nets))
        if ishandle(x)
            delete_line(x);
        end
    end
    arrayfun(@delete_block, one_d);
    for k = 1:numel(reconnect)
        for q = reconnect{k}(2:end)
            add_line(mdl, reconnect{k}(1), q, 'autorouting', 'on');
        end
    end
    for k = 1:size(pairs, 1)
        add_line(mdl, pairs(k, 1), pairs(k, 2), 'autorouting', 'on');
    end
    removed = size(pairs, 1);

    for ref = {lib.interface, lib.source, lib.reference}
        assert(isempty(find_ref(ref{1})), 'gs3dx:directDrive', ...
            'Postcondition: %s still contains %s', mdl, ref{1});
    end
    g = local_graph(mdl);
    for q = cell2mat(keys(expected))
        actual = sort(reshape(local_partners(g, q, []), 1, []));
        want   = sort(reshape(expected(q), 1, []));
        assert(isequal(actual, want) || (isempty(actual) && isempty(want)), 'gs3dx:directDrive', ...
            'Postcondition: port on %s now reaches [%s], expected [%s]', get_param(q, 'Parent'), ...
            local_names(actual), local_names(want));
    end
end

function g = local_graph(mdl)
% Physical ports of MDL's top-level blocks with their owning block and net id
% (0 = unconnected), plus every line segment and its net id.
    g.line = reshape(find_system(mdl, 'FindAll', 'on', 'SearchDepth', 1, 'Type', 'line'), 1, []);
    parent = 1:numel(g.line);                     % union-find over segments
    function r = root(i)
        r = i;
        while parent(r) ~= r
            r = parent(r);
        end
    end
    for i = 1:numel(g.line)
        kids = get_param(g.line(i), 'LineChildren');
        for c = reshape(kids, 1, [])
            j = find(g.line == c, 1);
            if ~isempty(j)
                parent(root(j)) = root(i);
            end
        end
    end
    g.line_net = arrayfun(@(i) g.line(root(i)), 1:numel(g.line));

    blocks = find_system(mdl, 'SearchDepth', 1, 'Type', 'Block');
    g.port = []; g.owner = []; g.net = [];
    for k = 1:numel(blocks)
        ph = get_param(blocks{k}, 'PortHandles');
        for q = [ph.LConn ph.RConn]
            l = get_param(q, 'Line');
            n = 0;
            if l > 0
                n = g.line_net(g.line == l);
            end
            g.port(end+1)  = q;
            g.owner(end+1) = get_param(blocks{k}, 'Handle');
            g.net(end+1)   = n;
        end
    end
end

function far = local_partners(g, p, excluded_owners)
% Ports on the net of port P, excluding P and ports owned by EXCLUDED_OWNERS.
    n = g.net(g.port == p);
    far = [];
    if n > 0
        far = g.port(g.net == n & g.port ~= p & ~ismember(g.owner, excluded_owners));
    end
end

function far = local_point_to_point(g, blk, candidates)
% Ports owned by CANDIDATES on two-port nets of BLK (the shared ground net,
% which reaches every source, is never two-port).
    me = get_param(blk, 'Handle');
    far = [];
    for n = unique(g.net(g.owner == me & g.net > 0))
        on_net = g.net == n;
        if nnz(on_net) == 2 && any(ismember(g.owner(on_net), candidates))
            far(end+1) = g.port(on_net & g.owner ~= me); %#ok<AGROW>
        end
    end
end

function s = local_names(ports)
    s = strjoin(arrayfun(@(x) strrep(get_param(get_param(x, 'Parent'), 'Name'), newline, ' '), ...
        ports, 'UniformOutput', false), ', ');
end

function ref = local_ref(blk)
% Library path with line breaks in block names normalised to spaces.
    ref = strrep(get_param(blk, 'ReferenceBlock'), newline, ' ');
end
