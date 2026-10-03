function ref = gs3dx_upper_body_reference(ik, opts)
%GS3DX_UPPER_BODY_REFERENCE  Upper-body joint references from a whole-body IK (#10979).
%
%   REF = GS3DX_UPPER_BODY_REFERENCE(IK) turns the KinematicsSolver joint
%   values of a GS3DX_WHOLE_BODY_IK (on GS3DX_Fit) into the angles that the
%   upper-body '<J> Input Function' charts read, for the twelve joints
%   Spine, Torso, L/R Scap, L/R S(houlder), L/R E(lbow), L/R F(orearm) and
%   L/R W(rist):
%
%   * revolute (Torso, E, F) and universal (Spine, Scap, W) joints: the
%     primitive angles (Rz; Rx, Ry) in degrees, unwrapped to one continuous
%     branch;
%   * spherical (shoulders): the intrinsic X-Y-Z angles of the joint
%     rotation (GS3DX_XYZ_MAP, the convention of the charts' 'XYZ
%     Kinematics'), each frame on the 360-degree branch of the previous one.
%
%   Every angle is filtered (zero-phase Butterworth, order 4, CUTOFF_HZ) and
%   differentiated (gradient) for the rates.
%
%   Options: cutoff_hz (12); joint_variables (table(), legacy Fit only).
%   For Human or another renumbered model, pass the native KinematicsSolver
%   jointPositionVariables table from the capture/model-bound IK. Block-path
%   keys are resolved by GS3DX_JOINT_KEYS; numbered Fit IDs are never reused.
%   This pure helper does not load/save a model or qualify forward dynamics.
%   Caller must bind table/model geometry and source hashes before use.
%   REF fields:
%     .model, .frames, .t (s, from the first frame), .rate_hz, .cutoff_hz
%     .joints     struct array: .prefix ('LS', ...), .axes ('XYZ', 'XY' or
%                 ''), .ids (solver variables), .angle and .rate
%                 (numel(axes) x frames, deg and deg/s; one row for '')
%     .start      struct of <prefix>StartPosition/Velocity[<axis>] at the
%                 first frame, the model-workspace start variables

    arguments
        ik (1,1) struct
        opts.cutoff_hz (1,1) double {mustBePositive} = 12
        opts.joint_variables table = table()
    end
    required={'model','frames','t','status','joint_ids','joint'};
    assert(all(isfield(ik,required)),'gs3dx:ubref','Missing IK reference fields');
    model=string(ik.model);
    assert(isscalar(model)&&~ismissing(model)&&strlength(model)>0,'gs3dx:ubref','Invalid model name');
    frames = ik.frames;
    n = numel(frames);
    assert(isnumeric(frames)&&isreal(frames)&&isvector(frames)&&n>=16&& ...
        all(isfinite(frames(:)))&&all(frames(:)>=1)&&all(frames(:)==fix(frames(:)))&& ...
        all(diff(frames(:))>0)&&all(diff(frames(:))==frames(2)-frames(1)), ...
        'gs3dx:ubref','IK frames must be positive integer, increasing and evenly spaced');
    assert(isnumeric(ik.status)&&isreal(ik.status)&&isvector(ik.status)&&numel(ik.status)==n&& ...
        all(isfinite(ik.status(:)))&&all(ik.status(:)>=1),'gs3dx:ubref','Invalid IK status');
    assert(isnumeric(ik.t)&&isreal(ik.t)&&isvector(ik.t)&&numel(ik.t)==n&& ...
        all(isfinite(ik.t(:)))&&all(diff(ik.t(:))>0),'gs3dx:ubref','Invalid physical time grid');
    t = ik.t(:).' - ik.t(1);
    dt=diff(t);assert(max(abs(dt-mean(dt)))<=1e-6*mean(dt), ...
        'gs3dx:ubref','Physical reference grid must be uniformly spaced');
    rate = (n - 1) / t(end);
    input_ids=string(ik.joint_ids(:));
    assert(~isempty(input_ids)&&all(~ismissing(input_ids))&&all(strlength(input_ids)>0)&& ...
        numel(unique(input_ids))==numel(input_ids),'gs3dx:ubref','Invalid or repeated joint IDs');
    assert(isnumeric(ik.joint)&&isreal(ik.joint)&&isequal(size(ik.joint),[numel(input_ids),n])&& ...
        all(isfinite(ik.joint),'all'),'gs3dx:ubref','Invalid native joint trajectory');
    assert(isfinite(opts.cutoff_hz)&&opts.cutoff_hz < rate / 2, 'gs3dx:ubref', ...
        'Cutoff above Nyquist (%g Hz)', rate / 2);
    jp=opts.joint_variables;binding_keys=strings(0,1);binding_ids=binding_keys;
    if isempty(jp)
        assert(model=="GS3DX_Fit",'gs3dx:ubref', ...
            'Renumbered models require an explicit native joint-variable table');
    else
        assert(all(ismember({'ID','BlockPath','Unit'},jp.Properties.VariableNames)), ...
            'gs3dx:ubref','Binding requires native ID, BlockPath and Unit columns');
        binding_ids=string(jp.ID);paths=string(jp.BlockPath);native_units=string(jp.Unit);
        assert(iscolumn(binding_ids)&&iscolumn(paths)&&iscolumn(native_units)&& ...
            numel(unique(binding_ids))==height(jp)&&all(~ismissing([binding_ids;paths;native_units]))&& ...
            all(startsWith(paths,model+"/"))&&isequal(sort(binding_ids),sort(input_ids)), ...
            'gs3dx:ubref','Binding must exactly identify this model and trajectory');
        try
            [binding_keys,shared_ids]=gs3dx_joint_keys(char(model),jp);
        catch binding_error
            error('gs3dx:ubref','Invalid native key binding: %s',binding_error.message);
        end
        assert(isequal(shared_ids,binding_ids),'gs3dx:ubref','Native ID mapping differs');
    end
    [b, a] = butter(4, opts.cutoff_hz / (rate / 2));

    ref = struct('model', ik.model, 'frames', frames, 't', t, 'rate_hz', rate, ...
        'cutoff_hz', opts.cutoff_hz);
    spec = gs3dx_upper_body_joints();
    for k = 1:numel(spec)
        j.prefix = spec(k).prefix;
        j.axes = spec(k).axes;
        j.ids = local_ids(ik, spec(k).id, j.axes, isempty(binding_keys));
        if ~isempty(binding_keys)
            suffix=extractAfter(string(j.ids),".");wanted=string(spec(k).block)+"|"+suffix;
            [found,rows]=ismember(wanted,binding_keys);
            assert(all(found),'gs3dx:ubref','Missing native upper joint primitives');
            expected_units=repmat("deg",size(rows));
            if numel(j.axes)==3,expected_units(1:3)="1";end
            assert(isequal(reshape(native_units(rows),size(rows)),expected_units), ...
                'gs3dx:ubref','Incorrect native upper joint units');
            j.ids=reshape(cellstr(binding_ids(rows)),size(j.ids));
        end
        raw = local_angles(ik, j);
        j.angle = filtfilt(b, a, raw.').';
        j.rate = gradient(j.angle, 1 / rate);
        joints(k) = j; %#ok<AGROW> twelve joints
    end
    ref.joints = joints;
    ref.start = local_start(joints);
end

function ids = local_ids(ik, j, axes, validate_ids)
    switch numel(axes)
        case 0
            ids = {[j '.Rz.q']};
        case 2
            ids = {[j '.Rx.q'], [j '.Ry.q']};
        otherwise
            ids = strcat(j, {'.S.ax_x', '.S.ax_y', '.S.ax_z', '.S.q'});
    end
    if validate_ids
        missing = setdiff(ids, ik.joint_ids);
        assert(isempty(missing), 'gs3dx:ubref', 'The IK has no joint variable %s', strjoin(missing, ', '));
    end
end

function ang = local_angles(ik, j)
    [~, rows] = ismember(j.ids, ik.joint_ids);
    v = ik.joint(rows, :);
    if numel(j.axes) < 3   % one continuous branch: the IK may wrap an angle to (-180, 180]
        ang = rad2deg(unwrap(deg2rad(v), [], 2));
        return
    end
    ang = zeros(3, size(v, 2));
    prev = zeros(3, 1);
    z = zeros(3, 1);
    for f = 1:size(v, 2)
        axis = v(1:3, f) / norm(v(1:3, f));
        Q = [cosd(v(4, f) / 2); sind(v(4, f) / 2) * axis];
        [~, prev] = gs3dx_xyz_map(Q, z, z, z, z, prev);
        ang(:, f) = prev;
    end
end

function s = local_start(joints)
    s = struct();
    for j = joints
        if isempty(j.axes)
            s.([j.prefix 'StartPosition']) = j.angle(1, 1);
            s.([j.prefix 'StartVelocity']) = j.rate(1, 1);
            continue
        end
        for a = 1:numel(j.axes)
            s.([j.prefix 'StartPosition' j.axes(a)]) = j.angle(a, 1);
            s.([j.prefix 'StartVelocity' j.axes(a)]) = j.rate(a, 1);
        end
    end
end
