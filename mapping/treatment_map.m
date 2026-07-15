function map = treatment_map(action, map, varargin)
%TREATMENT_MAP Accumulate observations and affected detections into a grid.
%   map = treatment_map("init", [], cfg) creates a treatment grid.
%   map = treatment_map("update", map, world_xy, class_probs, timestamp, cfg)
%   map = treatment_map("update", ..., footprint_xy) also marks all cells
%   inside the camera footprint as observed, independent of class prediction.
%   map = treatment_map("zones", map, cfg) extracts connected affected zones.

switch string(action)
    case "init"
        cfg = varargin{1};
        map = init_map(cfg);
    case "update"
        world_xy = varargin{1};
        class_probs = varargin{2};
        timestamp = varargin{3};
        cfg = varargin{4};
        footprint_xy = [];
        if numel(varargin) >= 5
            footprint_xy = varargin{5};
        end
        map = update_map(map, world_xy, class_probs, timestamp, cfg, footprint_xy);
    case "zones"
        cfg = varargin{1};
        map.zones = extract_zones(map, cfg);
    otherwise
        error("treatment_map:UnknownAction", "Unknown action: %s", action);
end
end

function map = init_map(cfg)
map = struct();
map.resolution_m = cfg.map.resolution_m;
map.field_length_m = cfg.field.length_m;
map.field_width_m = cfg.field.width_m;
map.log_odds = zeros(cfg.map.height_cells, cfg.map.width_cells);
map.observed = false(cfg.map.height_cells, cfg.map.width_cells);
map.first_detection_time = nan(cfg.map.height_cells, cfg.map.width_cells);
map.last_update_time = nan(cfg.map.height_cells, cfg.map.width_cells);
map.observation_count = zeros(cfg.map.height_cells, cfg.map.width_cells);
map.sprayed = false(cfg.map.height_cells, cfg.map.width_cells);
map.zones = struct([]);
end

function map = update_map(map, world_xy, class_probs, timestamp, cfg, footprint_xy)
if ~isfield(map, "observed")
    map.observed = map.observation_count > 0;
end

if ~isempty(footprint_xy)
    map = mark_observed_footprint(map, footprint_xy, timestamp);
elseif ~isempty(world_xy)
    map = mark_observed_points(map, world_xy, timestamp);
end

if isempty(world_xy) || isempty(class_probs)
    return;
end

affected_prob = max(class_probs(:, cfg.model.affected_class_ids), [], 2);
valid = all(isfinite(world_xy), 2) & affected_prob >= cfg.model.confidence_threshold;
[rows, cols] = world_to_grid(map, world_xy(valid, :));

for k = 1:numel(rows)
    r = rows(k);
    c = cols(k);
    map.log_odds(r, c) = saturate_log_odds(map.log_odds(r, c) + cfg.map.log_odds_hit, cfg);
    if isnan(map.first_detection_time(r, c))
        map.first_detection_time(r, c) = timestamp;
    end
    map.last_update_time(r, c) = timestamp;
end
end

function map = mark_observed_footprint(map, footprint_xy, timestamp)
footprint_xy = footprint_xy(all(isfinite(footprint_xy), 2), :);
if size(footprint_xy, 1) < 3
    return;
end
[x_centers, y_centers] = grid_centers(map);
inside = inpolygon(x_centers, y_centers, footprint_xy(:, 1), footprint_xy(:, 2));
map.observed = map.observed | inside;
map.observation_count(inside) = map.observation_count(inside) + 1;
map.last_update_time(inside) = timestamp;
end

function map = mark_observed_points(map, world_xy, timestamp)
[rows, cols] = world_to_grid(map, world_xy);
for k = 1:numel(rows)
    r = rows(k);
    c = cols(k);
    map.observed(r, c) = true;
    map.observation_count(r, c) = map.observation_count(r, c) + 1;
    map.last_update_time(r, c) = timestamp;
end
end

function [rows, cols] = world_to_grid(map, world_xy)
if isempty(world_xy)
    rows = [];
    cols = [];
    return;
end
cols_all = floor(world_xy(:, 1) ./ map.resolution_m) + 1;
rows_all = floor(world_xy(:, 2) ./ map.resolution_m) + 1;
inside = rows_all >= 1 & rows_all <= size(map.log_odds, 1) & ...
         cols_all >= 1 & cols_all <= size(map.log_odds, 2);
rows = rows_all(inside);
cols = cols_all(inside);
end

function [x_centers, y_centers] = grid_centers(map)
[cols, rows] = meshgrid(1:size(map.log_odds, 2), 1:size(map.log_odds, 1));
x_centers = (cols - 0.5) * map.resolution_m;
y_centers = (rows - 0.5) * map.resolution_m;
end

function value = saturate_log_odds(value, cfg)
min_value = -10;
max_value = 10;
if isfield(cfg.map, "log_odds_min")
    min_value = cfg.map.log_odds_min;
end
if isfield(cfg.map, "log_odds_max")
    max_value = cfg.map.log_odds_max;
end
value = max(min_value, min(max_value, value));
end

function zones = extract_zones(map, cfg)
prob = 1 ./ (1 + exp(-map.log_odds));
binary = prob >= cfg.map.min_probability_for_zone;
cc = bwconncomp(binary, 8);
stats = regionprops(cc, prob, "Area", "WeightedCentroid", "MaxIntensity", "PixelIdxList");
zones = struct("id", {}, "centroid_xy", {}, "area_m2", {}, "probability", {}, "first_detection_time", {});
for i = 1:numel(stats)
    area_m2 = stats(i).Area * map.resolution_m^2;
    if area_m2 < cfg.map.min_zone_area_m2
        continue;
    end
    centroid_px = stats(i).WeightedCentroid;
    centroid_xy = [(centroid_px(1) - 0.5) * map.resolution_m, (centroid_px(2) - 0.5) * map.resolution_m];
    first_time = min(map.first_detection_time(stats(i).PixelIdxList), [], "omitnan");
    zones(end + 1) = struct("id", numel(zones) + 1, "centroid_xy", centroid_xy, ...
        "area_m2", area_m2, "probability", stats(i).MaxIntensity, ...
        "first_detection_time", first_time); %#ok<AGROW>
end
end
