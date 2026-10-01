function on = gs3dx_balance_reference(ws, ref, com_offset, com_ref, err_id)
%GS3DX_BALANCE_REFERENCE  Write the centre-of-mass reference of a balance model (#10979).
%
%   ON = GS3DX_BALANCE_REFERENCE(WS, REF, COM_OFFSET, COM_REF, ERR_ID)
%   writes BalanceOn, BalanceCOMRef and BalanceCOMRate into the model
%   workspace WS of a GS3DX_FitBalance-family model built from the leg
%   reference REF (GS3DX_LEG_REFERENCE).
%
%   The reference is COM_REF (3 x frames, World, m) when given, else the
%   reference pelvis pose carrying COM_OFFSET (3 x frames, pelvis frame, m;
%   GS3DX_BALANCE_COM_OFFSET).  With neither, the reference is the pelvis
%   path and ON (BalanceOn) is 0.  BalanceCOMRate is its time gradient on
%   LegReferenceTime.  Errors use identifier ERR_ID.

    arguments
        ws
        ref (1,1) struct
        com_offset double
        com_ref double
        err_id (1,:) char
    end
    assert(isempty(com_offset) || isempty(com_ref), err_id, 'Give COM_OFFSET or COM_REF, not both');
    T = ws.getVariable('LegReferenceTime');
    n = numel(T);
    on = ~isempty(com_offset) || ~isempty(com_ref);
    if ~isempty(com_ref)
        assert(isequal(size(com_ref), [3 n]) && all(isfinite(com_ref), 'all'), err_id, ...
            'COM_REF must be finite, 3 x %d', n);
        com = com_ref;
    else
        offset = zeros(3, n);
        if on
            assert(isequal(size(com_offset), [3 n]), err_id, 'COM_OFFSET must be 3 x %d', n);
            offset = com_offset;
        end
        com = ref.pelvis_p + squeeze(pagemtimes(ref.pelvis_R, reshape(offset, 3, 1, n)));
    end
    assignin(ws, 'BalanceOn', double(on));
    assignin(ws, 'BalanceCOMRef', com);
    rate = zeros(3, n);
    for r = 1:3
        rate(r, :) = gradient(com(r, :), T(:).');
    end
    assignin(ws, 'BalanceCOMRate', rate);
end
