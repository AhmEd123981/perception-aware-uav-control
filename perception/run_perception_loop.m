function [treatment, frame_log] = run_perception_loop(uav_state, field_model, treatment, t, frame_idx, cfg)
%RUN_PERCEPTION_LOOP Capture, segment, project, and update treatment map.
%   The default runtime uses a persistent Python ONNX Runtime worker so the
%   model session is loaded once per MATLAB simulation instead of once per
%   frame. The first three frames are marked as warm-up for timing metrics.

frame_log = default_frame_log(frame_idx);
if ~isfield(cfg, "enabled") || ~cfg.enabled
    return;
end

fps = cfg.camera.fps;
next_capture_time = (max(frame_idx, 1) - 1) / fps;
if t + 1e-9 < next_capture_time
    return;
end

root_dir = project_root();
exchange_root = fullfile(char(cfg.output_root), "exchange");
if ~exist(exchange_root, "dir")
    mkdir(exchange_root);
end
% Reuse a single per-run exchange location instead of leaking a fresh
% tempname() sub-directory for every frame. The old code left hundreds of
% tp<uuid> folders under results/perception_logs/*/exchange/. The MATLAB<->Python
% inference handshake is synchronous (write image -> infer -> read mask), so one
% image/mask pair overwritten each frame is sufficient. Per-frame RGB frames are
% still archived separately by camera_simulator when cfg.save_frames is true.
frame_dir = exchange_root;

frame = camera_simulator(uav_state, field_model, cfg, t, frame_idx);
image_path = fullfile(frame_dir, "frame_current.png");
mask_path = fullfile(frame_dir, "frame_current_mask.png");
imwrite(frame.rgb, image_path);

timer = tic;
runtime = string(get_nested_or(cfg, ["model", "runtime"], "python_onnx_worker"));
if runtime == "python_onnx_worker"
    worker = onnx_worker("start", cfg);
    response = onnx_worker("infer", worker, image_path, mask_path, cfg.model.input_size(1));
    command_output = response.message;
else
    [status, command_output] = run_system_inference(root_dir, image_path, mask_path, cfg);
    if status ~= 0
        error("run_perception_loop:InferenceFailed", ...
            "Python inference failed with status %d:\n%s", status, command_output);
    end
end
inference_time_ms = toc(timer) * 1000.0;

mask_python = imread(mask_path);
if ndims(mask_python) > 2
    mask_python = mask_python(:, :, 1);
end
mask_matlab = double(mask_python) + 1;

affected_ids = cfg.model.affected_class_ids(:)';
affected = ismember(mask_matlab, affected_ids);
[v, u] = find(affected);
pixel_uv = [u, v];

max_pixels = get_nested_or(cfg, ["model", "max_projected_pixels"], 20000);
if size(pixel_uv, 1) > max_pixels
    sample_idx = round(linspace(1, size(pixel_uv, 1), max_pixels));
    pixel_uv = pixel_uv(sample_idx, :);
end

world_xy = pixel_to_world(pixel_uv, frame.K, frame.R_world_cam, frame.t_world_cam, cfg.field.ground_z_m);
class_probs = one_hot_probabilities(mask_matlab, pixel_uv, numel(cfg.model.classes));
footprint_xy = camera_footprint_corners(frame, cfg);
treatment = treatment_map("update", treatment, world_xy, class_probs, t, cfg, footprint_xy);

frame_log.captured = true;
frame_log.frame_idx = frame_idx;
frame_log.next_frame_idx = frame_idx + 1;
frame_log.timestamp = t;
frame_log.exchange_dir = frame_dir;
frame_log.image_path = image_path;
frame_log.mask_path = mask_path;
frame_log.inference_time_ms = inference_time_ms;
frame_log.is_warmup = frame_idx <= 3;
frame_log.num_affected_pixels = size(pixel_uv, 1);
frame_log.num_projected_pixels = sum(all(isfinite(world_xy), 2));
frame_log.footprint_xy = footprint_xy;
frame_log.command = char(runtime);
frame_log.command_output = command_output;
end

function [status, command_output] = run_system_inference(root_dir, image_path, mask_path, cfg)
model_path = resolve_path(root_dir, cfg.model.onnx_path);
infer_script = fullfile(root_dir, "deep_learning", "infer.py");
python_exe = "python";
if isfield(cfg.model, "python_executable")
    python_exe = cfg.model.python_executable;
end
image_size = cfg.model.input_size(1);
cmd = sprintf("%s %s --image %s --model %s --output-mask %s --image-size %d", ...
    quote_path(python_exe), quote_path(infer_script), quote_path(image_path), ...
    quote_path(model_path), quote_path(mask_path), image_size);
[status, command_output] = system(cmd);
end

function frame_log = default_frame_log(frame_idx)
frame_log = struct();
frame_log.captured = false;
frame_log.frame_idx = frame_idx;
frame_log.next_frame_idx = frame_idx;
frame_log.timestamp = NaN;
frame_log.exchange_dir = "";
frame_log.image_path = "";
frame_log.mask_path = "";
frame_log.inference_time_ms = NaN;
frame_log.is_warmup = false;
frame_log.num_affected_pixels = 0;
frame_log.num_projected_pixels = 0;
frame_log.footprint_xy = zeros(0, 2);
frame_log.command = "";
frame_log.command_output = "";
end

function footprint_xy = camera_footprint_corners(frame, cfg)
height = cfg.camera.resolution(1);
width = cfg.camera.resolution(2);
corners_uv = [1, 1; width, 1; width, height; 1, height];
footprint_xy = pixel_to_world(corners_uv, frame.K, frame.R_world_cam, frame.t_world_cam, cfg.field.ground_z_m);
end

function probs = one_hot_probabilities(mask_matlab, pixel_uv, num_classes)
num_pixels = size(pixel_uv, 1);
probs = zeros(num_pixels, num_classes);
if num_pixels == 0
    return;
end
linear_idx = sub2ind(size(mask_matlab), pixel_uv(:, 2), pixel_uv(:, 1));
class_ids = mask_matlab(linear_idx);
class_ids = max(1, min(num_classes, class_ids));
for i = 1:num_pixels
    probs(i, class_ids(i)) = 1.0;
end
end

function path_out = resolve_path(root_dir, path_in)
path_in = char(path_in);
if isfile(path_in)
    path_out = path_in;
else
    path_out = fullfile(root_dir, path_in);
end
end

function root_dir = project_root()
this_file = mfilename("fullpath");
root_dir = fileparts(fileparts(this_file));
end

function quoted = quote_path(path_in)
path_in = char(path_in);
quoted = ['"', strrep(path_in, '"', '\"'), '"'];
end

function value = get_nested_or(s, names, default_value)
value = default_value;
cursor = s;
for i = 1:numel(names)
    name = char(names(i));
    if isstruct(cursor) && isfield(cursor, name)
        cursor = cursor.(name);
    else
        return;
    end
end
value = cursor;
end
