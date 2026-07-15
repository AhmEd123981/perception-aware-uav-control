function cfg = perception_config()
%PERCEPTION_CONFIG Configuration for opt-in UAV perception extension.
%   cfg = perception_config() returns a deterministic configuration struct
%   for camera simulation, segmentation inference, treatment mapping, and
%   revisit planning. Existing controller comparisons should call this only
%   when config.perception.enabled is true.

cfg = struct();

% Global switch and reproducibility.
cfg.enabled = false;
cfg.seed = 2026;
cfg.output_root = fullfile("results", "perception_logs");
cfg.save_frames = true;
cfg.save_masks = true;

% Field geometry matches the agricultural thesis scenario.
cfg.field = struct();
cfg.field.length_m = 15.0;
cfg.field.width_m = 10.0;
cfg.field.ground_z_m = 0.0;
cfg.field.spray_altitude_m = 4.0;
cfg.field.origin_xy_m = [0.0, 0.0];

% Downward-facing RGB camera model.
cfg.camera = struct();
cfg.camera.name = "downward_rgb_pinhole";
cfg.camera.resolution = [720, 1280];       % [height, width] pixels
cfg.camera.horizontal_fov_deg = 60.0;
cfg.camera.vertical_fov_deg = 37.6;        % Derived for 16:9 sensor
cfg.camera.fps = 5.0;
cfg.camera.sensor_size_mm = [6.4, 3.6];    % 1/2.3 inch style sensor
cfg.camera.focal_length_mm = 5.54;
cfg.camera.optical_axis_body = [0; 0; 1];  % Body +z points down in camera frame convention here
cfg.camera.frame_stride = [];              % Computed from sim dt at runtime
cfg.camera.K = build_camera_intrinsics(cfg.camera);

% Segmentation model paths and runtime.
cfg.model = struct();
cfg.model.task = "semantic_segmentation";
cfg.model.architecture = "unet_resnet18";
% Real-data Agriculture-Vision model (2020). MATLAB is 1-based, the model
% outputs 0-based ids and run_perception_loop adds 1, so:
%   model 0 background -> matlab 1, 1 drydown -> 2, 2 weed -> 3, 3 double_plant -> 4.
% The "problems" to map/spray are drydown/weed/double_plant = matlab ids [2 3 4].
cfg.model.classes = ["background", "drydown", "weed_cluster", "double_plant"];
cfg.model.affected_class_ids = [2, 3, 4];
cfg.model.onnx_path = fullfile("deep_learning", "weights", "av_segmentation_2020.onnx");
cfg.model.runtime = "python_onnx_worker";  % Persistent ONNX Runtime bridge
cfg.model.confidence_threshold = 0.55;
cfg.model.input_size = [256, 256];

% Treatment map accumulator.
cfg.map = struct();
cfg.map.resolution_m = 0.25;
cfg.map.width_cells = ceil(cfg.field.length_m / cfg.map.resolution_m);
cfg.map.height_cells = ceil(cfg.field.width_m / cfg.map.resolution_m);
cfg.map.update_rule = "bayesian_log_odds";
cfg.map.log_odds_hit = 0.85;
cfg.map.log_odds_miss = -0.20;
cfg.map.log_odds_min = -10.0;
cfg.map.log_odds_max = 10.0;
cfg.map.min_probability_for_zone = 0.60;
cfg.map.min_zone_area_m2 = 0.50;

% Revisit planner and spray logging.
cfg.replanner = struct();
cfg.replanner.enabled = true;
cfg.replanner.phase = "post_coverage_revisit";
cfg.replanner.centroid_merge_radius_m = 0.75;
cfg.replanner.revisit_altitude_m = cfg.field.spray_altitude_m;
cfg.replanner.revisit_speed_mps = 1.0;
cfg.replanner.spray_radius_m = 0.40;
cfg.replanner.spray_dwell_s = 1.0;
% Extra hover time the reference holds over each zone BEFORE the spray-dwell
% clock can complete, so the controller has time to settle within spray_radius
% on arrival. Was implicitly 0 (reference held a zone exactly spray_dwell_s),
% which gave zero settling margin and caused brittle NaN latencies for
% controllers that arrive with overshoot (e.g. PID/SMC on the circular path).
cfg.replanner.spray_settle_s = 2.0;

end

function K = build_camera_intrinsics(camera)
%BUILD_CAMERA_INTRINSICS Create pinhole intrinsic matrix from FOV.
height = camera.resolution(1);
width = camera.resolution(2);
fx = (width / 2) / tand(camera.horizontal_fov_deg / 2);
fy = (height / 2) / tand(camera.vertical_fov_deg / 2);
cx = (width - 1) / 2;
cy = (height - 1) / 2;
K = [fx, 0, cx; 0, fy, cy; 0, 0, 1];
end
