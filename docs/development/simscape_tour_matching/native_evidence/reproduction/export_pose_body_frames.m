function export_pose_body_frames(repo,seed_path,pose_dirs,run_dir)
%EXPORT_POSE_BODY_FRAMES Reconstruct and verify native body frames for calibration.
% Pose directories contain disjoint frame-*.json outputs from attached_pose_probe.
% Every exported pose must reproduce its saved projected markers. No dynamics.
    arguments
        repo (1,1) string
        seed_path (1,1) string
        pose_dirs (:,1) string
        run_dir (1,1) string
    end
    assert(strcmp(version('-release'),'2025b'),'R2025b is required');
    assert(~isfolder(run_dir),'Preserve existing frame export');
    seed=jsondecode(fileread(seed_path));
    source=fullfile(repo,'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab');
    addpath(fullfile(source,'src/model'));addpath(genpath(fullfile(source,'src/functions')));
    addpath(fullfile(source,'motion_matching/shared'));
    load_system('GolfSwing3D_Kinetic');cleanup=onCleanup(@()close_system('GolfSwing3D_Kinetic',0)); %#ok<NASGU>
    ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
    assignin(ws,'UpperArmLength',seed.geometry_in(1));assignin(ws,'LowerArmLength',seed.geometry_in(2));
    [ks,schema]=build_golf_kinematics();
    addTargetVariables(ks,schema.q_ids);addOutputVariables(ks,schema.frame_ids);addOutputVariables(ks,schema.rotation_ids);
    mkdir(run_dir);copyfile(seed_path,fullfile(run_dir,'seed.json'));
    for directory=pose_dirs'
        files=dir(fullfile(directory,'frame-*.json'));assert(~isempty(files),'No pose files');
        for k=1:numel(files)
            path=fullfile(files(k).folder,files(k).name);pose=jsondecode(fileread(path));
            assert(strcmp(pose.source_sha256,seed.source_sha256) && isequal(pose.geometry_in(:),seed.geometry_in(:)), ...
                'Pose capture or geometry differs');
            assert(isequal(string(pose.coordinate_names(:)),schema.coordinate_names(:)),'Coordinate order differs');
            [values,flag,targets]=solve(ks,pose.q(:));
            assert(flag==1 && all(targets) && all(isfinite(values)),'Exported native pose must satisfy every target and constraint');
            n=numel(schema.frame_ids);origins=reshape(values(1:n),3,[])';
            rotations=intrinsic_xyz_to_rotm(reshape(values(n+1:end),3,[])');
            [found,selected]=ismember(string(pose.labels),string(seed.labels));assert(all(found));
            bodies=string(seed.body_names);[found,indices]=ismember(bodies(selected),string({schema.frames.name}));assert(all(found));
            projected=project_body_markers(origins,rotations,indices(:),seed.offsets_m(selected,:));
            parity=max(abs(projected-pose.marker_positions_m),[],'all');
            assert(parity<1e-8,'Export must reproduce the saved pose markers');
            result=struct('frame_index',pose.frame_index,'time_s',pose.time_s,'source_pose',path, ...
                'source_sha256',seed.source_sha256,'geometry_in',seed.geometry_in,'matlab',version, ...
                'body_names',string({schema.frames.name})','origins_m',origins, ...
                'rotations_world_from_body',permute(rotations,[3,1,2]), ...
                'saved_marker_max_difference_m',parity,'qualification','native kinematic frame export only');
            output=fullfile(run_dir,files(k).name);assert(~isfile(output),'Duplicate frame index');
            fid=fopen(output,'w');assert(fid~=-1);guard=onCleanup(@()fclose(fid));
            fprintf(fid,'%s',jsonencode(result,PrettyPrint=true));clear guard;
        end
    end
end
