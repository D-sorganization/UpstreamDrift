function geo = gs3dx_contact_geometry(mdl)
%GS3DX_CONTACT_GEOMETRY  Sole-contact geometry and contact-law parameters (#11709).
%
%   GEO = GS3DX_CONTACT_GEOMETRY(MDL) loads MDL (if not already loaded) and
%   reads the model-workspace variables GS3DX_BUILD_CONTACT assigns that
%   place the six sole-sphere contacts and the ground plane, and parametrise
%   the Spatial Contact Force law on every contact:
%     FootHeelOffset, FootLength, FootContactWidth, AnkleHeight,
%     FootContactRadius, GroundOffset, GroundRotation,
%     FootContactStiffness, FootContactDamping, FootContactTransitionWidth,
%     FootStaticFriction, FootDynamicFriction, FootFrictionCriticalVelocity
%
%   Shared by EXPORT_GS3DX_GRF_FIXTURE and GS3DX_RIGID_STANCE so the
%   workspace read happens in one place (DRY): the rigid-body stance proxy
%   reuses the exact values GS3DX_FullBodyContact simulates with, not a
%   recomputation from GS3DX_LEG_TABLE.
%
%   Precondition: GroundRotation's third column (plane normal) is within
%   1e-12 of world +Z (GroundNotLevel assertion) -- every tool downstream
%   assumes a level ground.

    arguments
        mdl (1,:) char
    end
    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    ws = get_param(mdl, 'ModelWorkspace');
    names = ["FootHeelOffset", "FootLength", "FootContactWidth", "AnkleHeight", ...
             "FootContactRadius", "GroundOffset", "GroundRotation", ...
             "FootContactStiffness", "FootContactDamping", "FootContactTransitionWidth", ...
             "FootStaticFriction", "FootDynamicFriction", "FootFrictionCriticalVelocity"];
    geo = struct();
    for n = names
        v = getVariable(ws, char(n));
        if isa(v, 'Simulink.Parameter'); v = v.Value; end
        geo.(n) = double(v);
    end
    assert(norm(geo.GroundRotation(:, 3) - [0; 0; 1]) < 1e-12, 'gs3dx:contact_geometry', ...
        'GroundNotLevel: the plane normal must be world +Z');
end
