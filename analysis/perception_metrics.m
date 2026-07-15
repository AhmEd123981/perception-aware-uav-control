function metrics = perception_metrics(controller_results, cfg)
%PERCEPTION_METRICS Compute perception/revisit metrics per controller.
%   coverage_percent is the percent of field grid cells observed at least
%   once by the camera footprint. False-alarm rate is detected affected area
%   outside the ground-truth affected patches divided by field area.

if isempty(controller_results)
    metrics = table();
    return;
end

num_runs = numel(controller_results);
controller = strings(num_runs, 1);
coverage_percent = nan(num_runs, 1);
affected_zone_recall = nan(num_runs, 1);
false_alarm_rate_per_m2 = nan(num_runs, 1);
mean_latency_s = nan(num_runs, 1);
median_latency_s = nan(num_runs, 1);
max_latency_s = nan(num_runs, 1);
missed_zones = nan(num_runs, 1);
mean_inference_time_ms = nan(num_runs, 1);
p95_inference_time_ms = nan(num_runs, 1);

for i = 1:num_runs
    run = controller_results(i);
    controller(i) = get_string_field(run, "controller", sprintf("run_%02d", i));
    treatment = get_field_or(run, "treatment", struct());
    zones = get_field_or(treatment, "zones", struct([]));
    spray_events = get_field_or(run, "spray_events", struct([]));
    frame_log = get_field_or(run, "frame_log", struct([]));

    coverage_percent(i) = compute_coverage_percent(treatment);
    det = compute_zone_detection(run, treatment, zones, cfg);
    affected_zone_recall(i) = det.recall;
    false_alarm_rate_per_m2(i) = compute_false_alarm_rate(run, treatment, cfg);

    latencies = extract_latencies(spray_events);
    if ~isempty(latencies)
        mean_latency_s(i) = mean(latencies, "omitnan");
        median_latency_s(i) = median(latencies, "omitnan");
        max_latency_s(i) = max(latencies, [], "omitnan");
    end

    % missed_zones is the number of ground-truth affected zones that were NOT
    % detected, so it is consistent with affected_zone_recall (recall == 1 =>
    % missed_zones == 0). When no ground truth is available to score detection
    % (det.n_missed is NaN), fall back to the treatment-completeness count of
    % detected zones that were never sprayed.
    if isnan(det.n_missed)
        detected_zone_count = numel(zones);
        if isstruct(spray_events) && ~isempty(spray_events) && isfield(spray_events, "zone_id")
            treated_zone_ids = unique([spray_events.zone_id]);
            n_treated = numel(treated_zone_ids);
        else
            n_treated = 0;
        end
        missed_zones(i) = max(0, detected_zone_count - n_treated);
    else
        missed_zones(i) = det.n_missed;
    end
    [mean_inference_time_ms(i), p95_inference_time_ms(i)] = compute_inference_times(frame_log);
end

metrics = table(controller, coverage_percent, affected_zone_recall, ...
    false_alarm_rate_per_m2, mean_latency_s, median_latency_s, max_latency_s, ...
    missed_zones, mean_inference_time_ms, p95_inference_time_ms);
end

function coverage = compute_coverage_percent(treatment)
if isfield(treatment, "observed") && ~isempty(treatment.observed)
    observed = treatment.observed;
elseif isfield(treatment, "observation_count") && ~isempty(treatment.observation_count)
    observed = treatment.observation_count > 0;
else
    coverage = NaN;
    return;
end
coverage = 100 * nnz(observed) / numel(observed);
end

function det = compute_zone_detection(run, treatment, detected_zones, cfg)
%COMPUTE_ZONE_DETECTION Score affected-zone detection against ground truth.
%   Returns a struct with fields:
%     recall   - fraction of ground-truth affected zones detected
%     n_gt     - number of ground-truth affected zones (NaN if unavailable)
%     n_missed - number of ground-truth zones NOT detected (NaN if unavailable)
%   When no usable ground truth exists, recall falls back to the legacy
%   treatment-based estimate and n_gt/n_missed stay NaN so the caller uses its
%   own treatment-completeness fallback for missed_zones.
det = struct("recall", NaN, "n_gt", NaN, "n_missed", NaN);
if isfield(run, "field_model") && isfield(run.field_model, "texture_mask") && ...
        ~isempty(run.field_model.texture_mask) && ...
        isfield(treatment, "log_odds") && ~isempty(treatment.log_odds)
    gt_mask = rasterize_texture_gt(run.field_model.texture_mask, treatment, cfg);
    [det.recall, det.n_gt, det.n_missed] = compute_mask_overlap_recall(gt_mask, treatment, cfg);
elseif isfield(run, "field_model") && isfield(run.field_model, "affected_patches") && ...
        isfield(treatment, "log_odds") && ~isempty(treatment.log_odds)
    [det.recall, det.n_gt, det.n_missed] = ...
        compute_patch_overlap_recall(run.field_model.affected_patches, treatment, cfg);
else
    det.recall = compute_zone_recall(run, treatment, detected_zones, cfg);
end
end

function recall = compute_zone_recall(run, treatment, detected_zones, cfg)
if isfield(run, "field_model") && isfield(run.field_model, "texture_mask") && ...
        ~isempty(run.field_model.texture_mask) && ...
        isfield(treatment, "log_odds") && ~isempty(treatment.log_odds)
    % Real Agriculture-Vision field: score against the real GT mask. A GT
    % connected component counts as recalled when any confident detected cell
    % overlaps it (robust to merged/adjacent detections).
    gt_mask = rasterize_texture_gt(run.field_model.texture_mask, treatment, cfg);
    recall = compute_mask_overlap_recall(gt_mask, treatment, cfg);
elseif isfield(run, "field_model") && isfield(run.field_model, "affected_patches") && ...
        isfield(treatment, "log_odds") && ~isempty(treatment.log_odds)
    recall = compute_patch_overlap_recall(run.field_model.affected_patches, treatment, cfg);
elseif isfield(run, "ground_truth_zones") && ~isempty(run.ground_truth_zones)
    gt = run.ground_truth_zones;
    if isempty(detected_zones)
        recall = 0;
        return;
    end
    detected_centroids = reshape([detected_zones.centroid_xy], 2, [])';
    matched = false(numel(gt), 1);
    for k = 1:numel(gt)
        gt_xy = gt(k).centroid_xy;
        matched(k) = any(vecnorm(detected_centroids - gt_xy, 2, 2) <= 0.75);
    end
    recall = mean(matched);
elseif isfield(run, "field_model") && isfield(run.field_model, "affected_patches")
    gt = patches_to_zones(run.field_model.affected_patches);
    recall = compute_zone_recall(struct("ground_truth_zones", gt), treatment, detected_zones, cfg);
elseif isfield(run, "spray_events") && ~isempty(detected_zones)
    treated_zone_ids = unique([run.spray_events.zone_id]);
    recall = numel(treated_zone_ids) / numel(detected_zones);
else
    recall = NaN;
end
end

function [recall, n_gt, n_missed] = compute_patch_overlap_recall(patches, treatment, cfg)
%COMPUTE_PATCH_OVERLAP_RECALL Count a patch as recalled when any confident
% affected grid cell overlaps its ground-truth ellipse. This avoids a false
% zero when adjacent detections merge into one large connected component and
% the component centroid no longer lies close to each individual patch.
n_gt = NaN;
n_missed = NaN;
if isempty(patches)
    recall = NaN;
    return;
end
prob = 1 ./ (1 + exp(-treatment.log_odds));
detected = prob >= cfg.map.min_probability_for_zone;
recalled = false(numel(patches), 1);
[cols, rows] = meshgrid(1:size(treatment.log_odds, 2), 1:size(treatment.log_odds, 1));
x = (cols - 0.5) * treatment.resolution_m;
y = (rows - 0.5) * treatment.resolution_m;
for i = 1:numel(patches)
    p = patches(i);
    patch_mask = ((x - p.center(1)).^2 / p.radius(1)^2) + ...
                 ((y - p.center(2)).^2 / p.radius(2)^2) <= 1;
    overlap_cells = nnz(detected & patch_mask);
    recalled(i) = overlap_cells > 0;
end
recall = mean(recalled);
n_gt = numel(patches);
n_missed = nnz(~recalled);
end

function rate = compute_false_alarm_rate(run, treatment, cfg)
if ~isfield(treatment, "log_odds") || isempty(treatment.log_odds)
    rate = NaN;
    return;
end
prob = 1 ./ (1 + exp(-treatment.log_odds));
detected = prob >= cfg.map.min_probability_for_zone;

if isfield(run, "ground_truth_affected_mask") && ~isempty(run.ground_truth_affected_mask)
    gt = logical(run.ground_truth_affected_mask);
elseif isfield(run, "field_model") && isfield(run.field_model, "texture_mask") && ...
        ~isempty(run.field_model.texture_mask)
    gt = rasterize_texture_gt(run.field_model.texture_mask, treatment, cfg);
elseif isfield(run, "field_model") && isfield(run.field_model, "affected_patches")
    gt = affected_mask_from_patches(treatment, run.field_model.affected_patches);
else
    gt = false(size(detected));
end

if ~isequal(size(gt), size(detected))
    gt = imresize(gt, size(detected), "nearest");
end
false_alarm_cells = detected & ~gt;
field_area_m2 = cfg.field.length_m * cfg.field.width_m;
rate = nnz(false_alarm_cells) * treatment.resolution_m^2 / field_area_m2;
end

function gt_mask = rasterize_texture_gt(texture_mask, treatment, cfg)
%RASTERIZE_TEXTURE_GT Rasterize the real GT texture mask into the treatment grid.
%   texture_mask stores av 0-based ids over the field bounds [0,L]x[0,W] m. Each
%   treatment cell centre is mapped to the nearest texture pixel (same linear
%   world->image mapping render_field uses via imref2d), converted to MATLAB ids
%   (+1), and flagged affected when it is a cfg.model.affected_class_ids class.
[ny, nx] = size(treatment.log_odds);
[trow, tcol] = size(texture_mask);
[cols, rows] = meshgrid(1:nx, 1:ny);
x = (cols - 0.5) * treatment.resolution_m;   % world x in [0, L]
y = (rows - 0.5) * treatment.resolution_m;   % world y in [0, W]
u = min(tcol, max(1, floor(x / cfg.field.length_m * tcol) + 1));
v = min(trow, max(1, floor(y / cfg.field.width_m * trow) + 1));
ids = double(texture_mask(sub2ind([trow, tcol], v, u))) + 1;  % 0-based av -> matlab
gt_mask = ismember(ids, cfg.model.affected_class_ids(:)');
end

function [recall, n_gt, n_missed] = compute_mask_overlap_recall(gt_mask, treatment, cfg)
%COMPUTE_MASK_OVERLAP_RECALL Fraction of GT affected components with a hit.
n_gt = NaN;
n_missed = NaN;
if ~any(gt_mask(:))
    recall = NaN;
    return;
end
prob = 1 ./ (1 + exp(-treatment.log_odds));
detected = prob >= cfg.map.min_probability_for_zone;
cc = bwconncomp(gt_mask);
recalled = false(cc.NumObjects, 1);
for i = 1:cc.NumObjects
    recalled(i) = any(detected(cc.PixelIdxList{i}));
end
recall = mean(recalled);
n_gt = cc.NumObjects;
n_missed = nnz(~recalled);
end

function gt = affected_mask_from_patches(treatment, patches)
[cols, rows] = meshgrid(1:size(treatment.log_odds, 2), 1:size(treatment.log_odds, 1));
x = (cols - 0.5) * treatment.resolution_m;
y = (rows - 0.5) * treatment.resolution_m;
gt = false(size(treatment.log_odds));
for i = 1:numel(patches)
    p = patches(i);
    d2 = ((x - p.center(1)).^2 / p.radius(1)^2) + ((y - p.center(2)).^2 / p.radius(2)^2);
    gt = gt | d2 <= 1;
end
end

function zones = patches_to_zones(patches)
zones = struct("id", {}, "centroid_xy", {});
for i = 1:numel(patches)
    zones(i).id = i; %#ok<AGROW>
    zones(i).centroid_xy = patches(i).center; %#ok<AGROW>
end
end

function latencies = extract_latencies(spray_events)
if isempty(spray_events)
    latencies = [];
elseif isfield(spray_events, "latency_s")
    latencies = [spray_events.latency_s];
else
    latencies = [];
end
latencies = latencies(isfinite(latencies));
end

function [mean_time, p95_time] = compute_inference_times(frame_log)
if isempty(frame_log) || ~isfield(frame_log, "inference_time_ms")
    mean_time = NaN;
    p95_time = NaN;
    return;
end
times = [frame_log.inference_time_ms];
if isfield(frame_log, "is_warmup")
    warmup = [frame_log.is_warmup];
    times = times(~warmup);
elseif numel(times) > 3
    times = times(4:end);
end
times = times(isfinite(times));
if isempty(times)
    mean_time = NaN;
    p95_time = NaN;
else
    mean_time = mean(times);
    times = sort(times(:));
    p95_idx = max(1, ceil(0.95 * numel(times)));
    p95_time = times(p95_idx);
end
end

function value = get_field_or(s, name, default_value)
if isstruct(s) && isfield(s, name)
    value = s.(name);
else
    value = default_value;
end
end

function value = get_string_field(s, name, default_value)
if isstruct(s) && isfield(s, name)
    value = string(s.(name));
else
    value = string(default_value);
end
end
