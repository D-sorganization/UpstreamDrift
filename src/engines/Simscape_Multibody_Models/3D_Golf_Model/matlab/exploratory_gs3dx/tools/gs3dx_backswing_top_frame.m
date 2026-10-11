function frame = gs3dx_backswing_top_frame(jc, strict)
%GS3DX_BACKSWING_TOP_FRAME  Existing Pelvis-Yaw Minimum Event Proxy.
% This follows the match export's existing phase convention. It is a phase
% proxy, not a measured anatomical event. STRICT excludes gap-filled pelvis
% samples and rejects absent orientation instead of using a time fraction.
    arguments
        jc (1,1) struct
        strict (1,1) logical = false
    end
    assert(isfield(jc,'impact_frame') && isnumeric(jc.impact_frame) && ...
        isscalar(jc.impact_frame) && isreal(jc.impact_frame) && ...
        isfinite(jc.impact_frame) && jc.impact_frame>=1 && jc.impact_frame==fix(jc.impact_frame), ...
        'gs3dx:ik:scapula','A positive integer capture impact frame is required');
    impact=jc.impact_frame;
    if isfield(jc,'pelvis_R')
        if strict
            assert(isnumeric(jc.pelvis_R) && isreal(jc.pelvis_R) && ...
                size(jc.pelvis_R,1)==3 && size(jc.pelvis_R,2)==3 && ...
                impact<=size(jc.pelvis_R,3),'gs3dx:ik:scapula','Invalid pelvis event orientation data');
        end
        yaw=squeeze(atan2(jc.pelvis_R(2,1,:),jc.pelvis_R(1,1,:)));
        valid=isfinite(yaw(:));
        if strict
            assert(isfield(jc,'gap') && isfield(jc.gap,'pelvis') && ...
                numel(jc.gap.pelvis)==numel(yaw),'gs3dx:ik:scapula','Missing pelvis event validity');
            valid=valid & ~jc.gap.pelvis(:);
        end
        search=1:min(impact,numel(yaw));search=search(valid(search));
        assert(~isempty(search),'gs3dx:ik:scapula','No measured pelvis event sample');
        [~,at]=min(yaw(search));frame=search(at);
    else
        assert(~strict,'gs3dx:ik:scapula','Protraction release requires measured pelvis event data or explicit top');
        frame=max(1,round(.7*impact));
    end
    if strict
        assert(frame>1 && frame<impact,'gs3dx:ik:scapula','Invalid backswing event proxy; provide explicit top');
    end
end
