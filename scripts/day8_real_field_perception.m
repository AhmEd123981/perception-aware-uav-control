function day8_real_field_perception()
%DAY8_REAL_FIELD_PERCEPTION End-to-end MATLAB test of the wired real model.
%   Flies a lawnmower survey over a REAL Agriculture-Vision field, captures a
%   downward camera frame at each waypoint, runs the wired av_segmentation_2020
%   ONNX model through the persistent worker, and accumulates a treatment map.
%   Proves step 4: MATLAB flight loop -> real field -> real model -> problem map.

addpath(genpath(pwd));
cfg = perception_config();          % now points at av_segmentation_2020.onnx (4 real classes)
cfg.enabled = true;
cfg.output_root = fullfile("results", "perception_logs", "day8_real_field");
if ~exist(cfg.output_root, "dir"); mkdir(cfg.output_root); end

fprintf("Model: %s\n", cfg.model.onnx_path);
fprintf("Classes: %s\n", strjoin(cfg.model.classes, ", "));
fprintf("Affected (problem) matlab-ids: %s\n", mat2str(cfg.model.affected_class_ids));

% Real field the drone flies over.
tex = fullfile("datasets", "real_field", "field_texture.png");
assert(isfile(tex), "real field texture missing: %s", tex);
mask = fullfile("datasets", "real_field", "field_mask.png");
assert(isfile(mask), ...
    "real field GT mask missing: %s (run: python deep_learning/av_field_flyover.py --save-field)", mask);
gt_mask = imread(mask);
if ndims(gt_mask) > 2; gt_mask = gt_mask(:, :, 1); end
field_model = struct();
field_model.texture_rgb = imread(tex);
field_model.texture_mask = uint8(gt_mask);        % av 0-based ids, aligned to texture
field_model.texture_bounds = [0.0, 15.0; 0.0, 10.0];

treatment = treatment_map("init", [], cfg);

% Lawnmower waypoints over the 15x10 m field at spray altitude.
alt = cfg.field.spray_altitude_m;
xs = linspace(2.0, 13.0, 6);
ys = linspace(1.5, 8.5, 4);
fps = cfg.camera.fps;
frame_idx = 1;
total_affected = 0;
for iy = 1:numel(ys)
    row = xs; if mod(iy, 2) == 0; row = fliplr(xs); end
    for x = row
        uav = struct();
        uav.position = [x; ys(iy); alt];
        uav.R_world_body = eye(3);
        uav.velocity = [0; 0; 0];
        t = (frame_idx - 1) / fps;
        [treatment, flog] = run_perception_loop(uav, field_model, treatment, t, frame_idx, cfg);
        if isfield(flog, "num_affected_pixels")
            total_affected = total_affected + flog.num_affected_pixels;
            fprintf("frame %02d @ (%.1f,%.1f): affected pixels=%d, inf=%.0f ms\n", ...
                frame_idx, x, ys(iy), flog.num_affected_pixels, flog.inference_time_ms);
        end
        frame_idx = frame_idx + 1;
    end
end

treatment = treatment_map("zones", treatment, cfg);
nz = 0;
if isfield(treatment, "zones"); nz = numel(treatment.zones); end
fprintf("\nTOTAL frames flown: %d | total affected pixels: %d | zones found: %d\n", ...
    frame_idx - 1, total_affected, nz);
save(fullfile(cfg.output_root, "treatment.mat"), "treatment");

% Score detections against the REAL GT mask using the exact same metric as the
% main pipeline (perception_metrics) and the Python flyover: a GT affected
% connected component counts as recalled when a confident detected cell overlaps
% it. This yields a MATLAB-side real-field recall consistent with av_field_flyover.
empty_spray = struct("zone_id", {}, "time_s", {}, "position", {}, "latency_s", {}, "dwell_s", {});
payload = struct("controller", "real_field", "treatment", treatment, ...
    "spray_events", empty_spray, "frame_log", struct([]), "field_model", field_model);
real_metrics = perception_metrics(payload, cfg);
fprintf("Real-GT affected-zone recall: %.3f | coverage: %.1f %% | false alarm: %.5f m2/m2\n", ...
    real_metrics.affected_zone_recall(1), real_metrics.coverage_percent(1), ...
    real_metrics.false_alarm_rate_per_m2(1));
writetable(real_metrics, fullfile(cfg.output_root, "real_field_metrics.csv"));

% Save a treatment probability map figure if the field is available.
try
    if isfield(treatment, "log_odds")
        prob = 1 ./ (1 + exp(-treatment.log_odds));
        f = figure("Visible", "off");
        imagesc(prob'); axis image; colorbar; title("Detected problem probability (real field)");
        xlabel("field x cells"); ylabel("field y cells");
        exportgraphics(f, fullfile(cfg.output_root, "treatment_probability.png"), "Resolution", 150);
        close(f);
    end
catch e
    fprintf("map plot skipped: %s\n", e.message);
end

onnx_worker("stop");
fprintf("DONE_DAY8\n");
end
