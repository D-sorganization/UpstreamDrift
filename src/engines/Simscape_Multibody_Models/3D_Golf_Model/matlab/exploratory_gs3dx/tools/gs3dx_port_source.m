function src = gs3dx_port_source(blk)
%GS3DX_PORT_SOURCE  Where to look up the port blocks of subsystem BLK.
%
%   SRC = GS3DX_PORT_SOURCE(BLK) is BLK itself, or, for a subsystem
%   reference, the (loaded) referenced subsystem: a freshly added reference
%   does not list its contents to FIND_SYSTEM until the model is updated.

    src = blk;
    if strcmp(get_param(blk, 'BlockType'), 'SubSystem')
        ref = get_param(blk, 'ReferencedSubsystem');
        if ~isempty(ref)
            load_system(ref);
            src = ref;
        end
    end
end
