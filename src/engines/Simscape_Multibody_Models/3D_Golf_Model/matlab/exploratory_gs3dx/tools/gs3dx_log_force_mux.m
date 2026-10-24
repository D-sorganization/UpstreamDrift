function gs3dx_log_force_mux(sys, ports, name)
%GS3DX_LOG_FORCE_MUX  Convert, mux and log N physical force ports as one signal (#11709).
%
%   GS3DX_LOG_FORCE_MUX(SYS, PORTS, NAME) adds one 'N'-unit PS-Simulink
%   Converter per physical outport in PORTS, muxes them in port order and
%   logs the muxed signal as NAME into a Terminator -- the same
%   convert/mux/log pattern GS3DX_BUILD_CONTACT uses for
%   'FootContactForces'.  Shared so GS3DX_RIGID_STANCE does not duplicate
%   it (DRY).
%
%   Precondition: every port in PORTS is a Simulink physical-signal
%   outport (e.g. a Spatial Contact Force block's total-force sensing
%   port) of a 3-vector (N*m or N) signal.

    arguments
        sys (1,:) char
        ports (1,:) double
        name (1,:) char
    end
    n = numel(ports);
    assert(n >= 1, 'gs3dx:log_force_mux', 'Precondition: PORTS must not be empty');
    mux = add_block('simulink/Signal Routing/Mux', [sys '/' name ' Mux'], 'Inputs', num2str(n), ...
        'Position', [2300 60 2305 60 + 30 * n]);
    for k = 1:n
        yy = 60 + 30 * (k - 1);
        c = add_block('nesl_utility/PS-Simulink Converter', sprintf('%s/%s Force %d', sys, name, k), ...
            'Unit', 'N', 'Position', [2200 yy 2230 yy + 20]);
        local_connect(sys, ports(k), local_port(c, 'LConn', 1));
        local_connect(sys, local_port(c, 'Outport', 1), local_port(mux, 'Inport', k));
    end
    t = add_block('simulink/Sinks/Terminator', [sys '/' name], 'Position', [2400 150 2420 170]);
    h = add_line(sys, local_port(mux, 'Outport', 1), local_port(t, 'Inport', 1), 'autorouting', 'on');
    set_param(h, 'Name', name);
    set_param(local_port(mux, 'Outport', 1), 'DataLogging', 'on');
end

function local_connect(sys, a, b)
    add_line(sys, a, b, 'autorouting', 'on');
end

function p = local_port(h, kind, n)
    ph = get_param(h, 'PortHandles');
    p = ph.(kind)(n);
end
