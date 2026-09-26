function gs3dx_redraw_nets(mdl, removed, joins)
%GS3DX_REDRAW_NETS  Delete blocks from physical nets and redraw the nets.
%
%   GS3DX_REDRAW_NETS(MDL, REMOVED, JOINS) edits the loaded diagram MDL:
%     1. deletes every line of every net that touches a block in REMOVED
%        (block handles) and deletes those blocks;
%     2. reconnects the remaining ports of each such net to each other;
%     3. adds a line for each row [p q] of JOINS (physical port handles of
%        blocks that are not removed).
%   Deleting whole nets first is what keeps branched nets intact: deleting a
%   block only removes the segments attached to it.
%
%   Postcondition (asserted): among the remaining ports of the touched nets
%   and the JOINS ports, two ports share a net exactly when the old nets
%   plus JOINS connect them; nothing else is attached to those nets.

    arguments
        mdl (1,:) char
        removed (1,:) double
        joins (:,2) double = zeros(0, 2)
    end
    g = gs3dx_physical_graph(mdl);
    assert(all(ismember(joins(:), g.port)) && ~any(ismember(joins(:), ...
        g.port(ismember(g.owner, removed)))), 'gs3dx:redraw', ...
        'Precondition: JOINS must be physical ports of blocks that stay in %s', mdl);

    nets = unique(g.net(ismember(g.owner, removed) & g.net > 0));
    kept = g.port(ismember(g.net, nets) & ~ismember(g.owner, removed));
    watched = unique([kept, joins(:).']);
    % Expected partition: old nets among kept ports, merged by JOINS.
    label = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for q = watched
        n = g.net(g.port == q);
        if n == 0 || ~ismember(n, nets)
            n = q;                                 % its own (or untouched) net
        end
        label(q) = n;
    end
    for k = 1:size(joins, 1)
        a = label(joins(k, 1)); b = label(joins(k, 2));
        for q = watched
            if label(q) == b
                label(q) = a;
            end
        end
    end

    for x = g.line(ismember(g.line_net, nets))
        if ishandle(x)
            delete_line(x);
        end
    end
    arrayfun(@delete_block, removed);
    for n = nets
        chain = kept(g.net(ismember(g.port, kept)) == n);
        for q = chain(2:end)
            add_line(mdl, chain(1), q, 'autorouting', 'on');
        end
    end
    for k = 1:size(joins, 1)
        add_line(mdl, joins(k, 1), joins(k, 2), 'autorouting', 'on');
    end

    g = gs3dx_physical_graph(mdl);
    for q = watched
        now_net = g.net(g.port == q);
        want = watched(arrayfun(@(p) label(p) == label(q), watched));
        have = g.port(g.net == now_net & now_net > 0);
        if isempty(have)
            have = q;                              % unconnected port
        end
        assert(isequal(sort(have), sort(want)), 'gs3dx:redraw', ...
            'Postcondition: port on %s now shares a net with %d ports, expected %d', ...
            strrep(get_param(get_param(q, 'Parent'), 'Name'), newline, ' '), ...
            numel(have), numel(want));
    end
end
