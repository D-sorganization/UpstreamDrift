function attached_pose_probe(repo,run_dir,state_path,capture_path,frame_indices)
%ATTACHED_POSE_PROBE Diagnose connected-model marker feasibility at fixed geometry.
% Each saved sample is a local kinematic fit with fixed attachments, not a
% forward replay, global lower bound, anatomical fit or torque acceptance.
% Caller supplies one-based capture indices in strictly increasing order.
    arguments
        repo (1,1) string
        run_dir (1,1) string
        state_path (1,1) string
        capture_path (1,1) string
        frame_indices (:,1) double {mustBeInteger,mustBePositive,mustBeNonempty}
    end
    assert(strcmp(version('-release'),'2025b'),'R2025b is required');
    assert(all(diff(frame_indices)>0),'attached_pose_probe:order','Indices must increase.');
    assert(~isfolder(run_dir),'attached_pose_probe:existingRun','Preserve existing experiment.');
    seed=jsondecode(fileread(state_path)); capture=jsondecode(fileread(capture_path));
    assert(strcmp(seed.source_sha256,capture.source_sha256),'attached_pose_probe:capture','Capture identity differs.');
    assert(frame_indices(end)<=numel(capture.time_s),'attached_pose_probe:range','Index exceeds capture.');
    source=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
    addpath(fullfile(source,'src/model'));addpath(genpath(fullfile(source,'src/functions')));
    addpath(fullfile(source,'motion_matching/shared'));
    load_system('GolfSwing3D_Kinetic');
    cleanup=onCleanup(@()close_system('GolfSwing3D_Kinetic',0)); %#ok<NASGU>
    ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
    assignin(ws,'UpperArmLength',seed.geometry_in(1));assignin(ws,'LowerArmLength',seed.geometry_in(2));
    [ks,schema]=build_golf_kinematics();
    assert(isequal(string(seed.coordinate_names(:)),schema.coordinate_names(:)), ...
        'attached_pose_probe:coordinates','Coordinate identity differs.');
    mkdir(run_dir);
    copyfile(state_path,fullfile(run_dir,'initial_state.json'));
    copyfile(capture_path,fullfile(run_dir,'capture.json'));
    copyfile(mfilename('fullpath')+".m",fullfile(run_dir,'attached_pose_probe.m'));
    copyfile(which('fit_golf_pose_seed'),fullfile(run_dir,'fit_golf_pose_seed.m'));
    copyfile(which('select_capture_marker_frame'),fullfile(run_dir,'select_capture_marker_frame.m'));
    q=seed.q(:);
    for k=1:numel(frame_indices)
        index=frame_indices(k);
        [points,selected]=select_capture_marker_frame(capture,index,string(seed.labels(:)));
        bodies=string(seed.body_names(:));
        timer=tic;
        result=fit_golf_pose_seed(ks,schema,q,bodies(selected),points,seed.offsets_m(selected,:));
        result.elapsed_s=toc(timer);result.frame_index=index;result.time_s=capture.time_s(index);
        result.matlab=version;result.source_sha256=seed.source_sha256;
        result.geometry_in=seed.geometry_in;result.labels=seed.labels(selected);
        result.selected_label_indices=selected;result.valid_marker_count=numel(selected);
        result.initial_guess_q=q;result.frame_indices=frame_indices;
        path=fullfile(run_dir,sprintf('frame-%05d.json',index));
        fid=fopen(path,'w');assert(fid~=-1,'Cannot save incremental pose result');
        guard=onCleanup(@()fclose(fid));
        fprintf(fid,'%s',jsonencode(result,PrettyPrint=true));clear guard;
        q=result.q;
    end
end
