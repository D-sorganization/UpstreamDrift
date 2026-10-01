function report = gs3dx_build_golfer(info, opts)
%GS3DX_BUILD_GOLFER  Build GS3DX_Golfer: the contact model with typical segment masses.
%
%   REPORT = GS3DX_BUILD_GOLFER(INFO) copies GS3DX_FullBodyContact to
%   GS3DX_Golfer and replaces the upper-body masses (#11011).  The original
%   upper body carries literal masses in mixed units (kg, lbm, a density)
%   that add up to 77.6 kg, a full body on its own, so with the de Leva legs
%   the contact model weighs 109 kg (docs/DATA_AUDIT.md).  Here every body
%   solid gets its mass from one table, GS3DX_ANTHROPOMETRY(body_mass):
%
%   * The Golfer* model-workspace variables hold the upper-body segment
%     masses (kg); the leg variables ThighMass/ShankMass/FootMass are set
%     from the same table.  The drive files never set any of them.
%   * Each upper-body solid is switched to BasedOnType 'Mass' in kg with an
%     expression of those variables (SOLIDS below).  Inertia stays
%     CalculateFromGeometry, so it scales with the mass on the unchanged
%     geometry.
%   * Geometry, joints, drive, stance, contacts and servo are unchanged.
%     The start state is the contact model's.
%
%   The edit is parameter-only: the nonvirtual block count is asserted
%   unchanged, and REPORT.budget holds the compiled count.
%
%   Options: overwrite (false) replaces an existing GS3DX_Golfer;
%   body_mass (80 kg, the mass the legs are scaled for).
%
%   REPORT fields: .vars (the variables set), .solids (block, expression),
%   .equipment_mass (kg: club, grip parts, contact spheres; not body
%   segments), .expected_mass (body_mass + equipment_mass) and .budget.

    arguments
        info (1,1) struct
        opts.overwrite (1,1) logical = false
        opts.body_mass (1,1) double {mustBePositive} = 80
    end
    names = gs3dx_names();
    src = char(names.variants.contact);
    mdl = char(names.variants.golfer);
    gs3dx_copy_models({fullfile(info.models_dir, [src '.slx']), fullfile(info.models_dir, [mdl '.slx'])}, ...
        opts.overwrite, 'gs3dx:golfer');

    anthro = gs3dx_anthropometry(opts.body_mass);
    load_system(mdl);
    cleanup = onCleanup(@() close_system(mdl, 0));
    before = local_nonvirtual(mdl);

    ws = get_param(mdl, 'ModelWorkspace');
    vars = anthro.vars;
    for f = fieldnames(anthro.legs).'
        vars.(f{1}) = anthro.legs.(f{1});
    end
    for f = fieldnames(vars).'
        assignin(ws, f{1}, vars.(f{1}));
    end

    solids = local_solids();
    for k = 1:size(solids, 1)
        blk = [mdl '/' solids{k, 1}];
        assert(getSimulinkBlockHandle(blk) > 0, 'gs3dx:golfer', 'Solid not found: %s', blk);
        assert(strcmp(get_param(blk, 'InertiaType'), 'CalculateFromGeometry'), 'gs3dx:golfer', ...
            '%s does not compute its inertia from geometry', blk);
        set_param(blk, 'BasedOnType', 'Mass', 'Mass', solids{k, 2}, 'MassUnits', 'kg');
    end

    after = local_nonvirtual(mdl);
    assert(after == before, 'gs3dx:golfer', ...
        'Postcondition: the mass edit changed the block count (%d -> %d)', before, after);
    report.equipment_mass = local_equipment_mass(mdl);
    report.expected_mass = opts.body_mass + report.equipment_mass;
    report.budget = gs3dx_block_budget(mdl, compiled=true);
    reserve = 25;   % room for the sensors GS3DX_CONTACT_CHECK adds in memory
    assert(report.budget.compiled_total <= names.license_block_limit - reserve, 'gs3dx:golfer', ...
        'GS3DX_Golfer compiles to %d blocks, leaving no room for %d validation blocks', ...
        report.budget.compiled_total, reserve);
    gs3dx_save_model(mdl, info);
    close_system(mdl, 0);
    report.vars = vars;
    report.solids = solids;
end

function s = local_solids()
%LOCAL_SOLIDS  Every upper-body solid that carries mass, and its new mass (kg).
%   Club, grip parts and the massless joint/marker solids are unchanged.
    T = 'Hips and Torso Inputs/';
    s = {
        [T 'Head'],                    'GolferHeadMass'
        [T 'Neck'],                    'GolferNeckMass'
        [T 'LowerTorso'],              'GolferLowerTrunkMass'
        [T 'UpperTorsoBase'],          '0.2*GolferUpperTrunkMass'   % the original 0.2/0.8 length split
        [T 'UpperTorsoTop'],           '0.8*GolferUpperTrunkMass'
        'HubtoLS',                     'GolferShoulderMass'
        'HubtoRS',                     'GolferShoulderMass'
        'LUpperArm',                   'GolferUpperArmMass'
        'RUpperArm',                   'GolferUpperArmMass'
        'Left Forearm/LUpperForearm',  '0.5*GolferForearmMass'
        'Left Forearm/LLowerForearm',  '0.5*GolferForearmMass'
        'Right Forearm/RUpperForearm', '0.5*GolferForearmMass'
        'Right Forearm/RLowerForearm', '0.5*GolferForearmMass'
        'Grip/LHand',                  'GolferHandMass'
        'Grip/RHand',                  'GolferHandMass'
        };
end

function m = local_equipment_mass(mdl)
%LOCAL_EQUIPMENT_MASS  Club, grip parts and contact spheres: literal masses.
    blks = [find_system([mdl '/Club'], 'LookUnderMasks', 'all', 'Regexp', 'on', 'ReferenceBlock', 'sm_lib/Body Elements/.* Solid'); ...
            find_system([mdl '/Grip'], 'LookUnderMasks', 'all', 'Regexp', 'on', 'ReferenceBlock', 'sm_lib/Body Elements/.* Solid'); ...
            find_system([mdl '/Lower Body'], 'SearchDepth', 1, 'Regexp', 'on', 'Name', ' Sphere$')];
    blks = blks(~endsWith(blks, {'/LHand', '/RHand'}));
    units = struct('kg', 1, 'g', 1e-3, 'lbm', 0.45359237);
    m = 0;
    for k = 1:numel(blks)
        assert(strcmp(get_param(blks{k}, 'BasedOnType'), 'Mass'), 'gs3dx:golfer', ...
            'Equipment solid %s is not mass-based', blks{k});
        v = str2double(get_param(blks{k}, 'Mass'));
        assert(isfinite(v), 'gs3dx:golfer', 'Equipment solid %s has no literal mass', blks{k});
        m = m + v * units.(get_param(blks{k}, 'MassUnits'));
    end
end

function n = local_nonvirtual(mdl)
    n = numel(find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'LookInsideSubsystemReference', 'on', 'Virtual', 'off'));
end
