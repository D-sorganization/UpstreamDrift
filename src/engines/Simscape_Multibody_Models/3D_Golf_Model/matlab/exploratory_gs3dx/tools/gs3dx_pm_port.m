function p = gs3dx_pm_port(blk, name)
%GS3DX_PM_PORT  Physical port handle of subsystem BLK for its PMIOPort NAME.
%
%   P = GS3DX_PM_PORT(BLK, NAME) returns the LConn/RConn port handle of the
%   subsystem (or subsystem reference) BLK that belongs to the PMIOPort
%   named NAME inside it.  PMIOPorts are numbered per side, so the handle
%   is found from the port's side and its rank among that side's ports.
%   A subsystem reference is resolved through GS3DX_PORT_SOURCE.

    arguments
        blk
        name (1,:) char
    end
    pm = find_system(gs3dx_port_source(blk), 'SearchDepth', 1, 'BlockType', 'PMIOPort');
    side = get_param(pm, 'Side');
    mine = find(strcmp(get_param(pm, 'Name'), name));
    assert(isscalar(mine), 'gs3dx:port', 'No PMIOPort %s in %s', name, getfullname(blk));
    same = find(strcmp(side, side{mine}));
    order = str2double(get_param(pm(same), 'Port'));
    [~, rank] = sort(order);
    idx = find(same(rank) == mine);
    ph = get_param(blk, 'PortHandles');
    p = ph.([side{mine}(1) 'Conn'])(idx);
end
