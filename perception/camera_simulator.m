function frame = camera_simulator(uav_state, field_model, cfg, sim_time, frame_index)
%CAMERA_SIMULATOR Capture a synthetic downward RGB frame from UAV pose.
%   frame = camera_simulator(uav_state, field_model, cfg, sim_time, idx)
%   renders a pinhole camera view, returns image, pose metadata, intrinsics,
%   and file paths. The function is deterministic under cfg.seed.
%
%   Expected uav_state fields:
%     position: [3x1] world position [x;y;z] in meters
%     R_world_body: [3x3] body-to-world rotation matrix
%     velocity: [3x1] optional velocity for motion blur model

arguments
    uav_state struct
    field_model struct
    cfg struct
    sim_time (1,1) double
    frame_index (1,1) double {mustBeInteger, mustBeNonnegative}
end

camera = cfg.camera;
K = camera.K;

% Camera is rigidly mounted downward. Replace this with a calibrated
% extrinsic if the body/camera convention differs in the main simulator.
R_body_cam = [1 0 0; 0 -1 0; 0 0 -1];
R_world_cam = uav_state.R_world_body * R_body_cam;
t_world_cam = uav_state.position(:);

render_opts = struct();
render_opts.resolution = camera.resolution;
render_opts.K = K;
render_opts.R_world_cam = R_world_cam;
render_opts.t_world_cam = t_world_cam;
render_opts.ground_z_m = cfg.field.ground_z_m;
render_opts.apply_motion_blur = isfield(uav_state, "velocity") && norm(uav_state.velocity) > 0.5;
render_opts.seed = cfg.seed + frame_index;

[rgb, semantic_gt, world_footprint] = render_field(field_model, render_opts);

frame = struct();
frame.index = frame_index;
frame.timestamp = sim_time;
frame.rgb = rgb;
frame.semantic_gt = semantic_gt;
frame.K = K;
frame.R_world_cam = R_world_cam;
frame.t_world_cam = t_world_cam;
frame.world_footprint = world_footprint;
frame.pose.position = t_world_cam;
frame.pose.R_world_body = uav_state.R_world_body;
frame.pose.R_world_cam = R_world_cam;

if cfg.save_frames
    out_dir = fullfile(cfg.output_root, "frames");
    if ~exist(out_dir, "dir")
        mkdir(out_dir);
    end
    frame.image_path = fullfile(out_dir, sprintf("frame_%06d.png", frame_index));
    imwrite(rgb, frame.image_path);

    meta_path = fullfile(out_dir, sprintf("frame_%06d_pose.mat", frame_index));
    pose = frame.pose; %#ok<NASGU>
    timestamp = sim_time; %#ok<NASGU>
    K_save = K; %#ok<NASGU>
    save(meta_path, "pose", "timestamp", "K_save", "world_footprint");
    frame.metadata_path = meta_path;
end

end
