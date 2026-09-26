function g = gs3dx_physical_graph(mdl)
%GS3DX_PHYSICAL_GRAPH  Physical-connection nets of a diagram's top level.
%
%   G = GS3DX_PHYSICAL_GRAPH(MDL) returns, for the loaded diagram MDL:
%     .line      (1,L) handles of every line segment
%     .line_net  (1,L) net id per segment (the handle of its root segment)
%     .port      (1,P) physical port handles (LConn and RConn) of every block
%     .owner     (1,P) owning block handle per port
%     .net       (1,P) net id per port (0 = unconnected)
%
%   Nets are the connected components of all segments.  A branched net lists
%   its segments only on the trunk (LineChildren, which includes the trunk
%   itself), and branch segments report no parent or end ports, so walking
%   from a port's own segment is not enough.

    arguments
        mdl (1,:) char
    end
    g.line = reshape(find_system(mdl, 'FindAll', 'on', 'SearchDepth', 1, 'Type', 'line'), 1, []);
    parent = 1:numel(g.line);                     % union-find over segments
    function r = root(i)
        r = i;
        while parent(r) ~= r
            r = parent(r);
        end
    end
    for i = 1:numel(g.line)
        for c = reshape(get_param(g.line(i), 'LineChildren'), 1, [])
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
