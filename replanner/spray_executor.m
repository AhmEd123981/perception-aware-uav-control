function [treatment, spray_state, spray_events, latency_by_zone] = spray_executor(uav_state, treatment, plan, t, spray_state, cfg)
%SPRAY_EXECUTOR Log targeted spray events during revisit flight.
%   The executor is stateful through spray_state. Call it once per simulation
%   step during the revisit phase. A zone is sprayed after the UAV remains
%   within cfg.replanner.spray_radius_m for cfg.replanner.spray_dwell_s.
%
%   Opportunistic scheduling: every not-yet-sprayed zone is checked each step,
%   so a zone the controller cannot reach does NOT block the remaining zones.
%   (The old executor was strictly in-order and single-target: missing one zone
%   froze the whole queue, which produced all-or-nothing NaN treatment latency
%   for controllers that overshoot on aggressive trajectories.)

if nargin < 5 || isempty(spray_state)
    spray_state = initialize_spray_state(plan);
end

spray_events = spray_state.events;
latency_by_zone = spray_state.latency_by_zone;
if ~isfield(plan, "zones") || isempty(plan.zones)
    return;
end

position = extract_position(uav_state);
radius = cfg.replanner.spray_radius_m;
dwell_required = cfg.replanner.spray_dwell_s;

for zi = 1:numel(plan.zones)
    zone = plan.zones(zi);
    if spray_state.sprayed(zone.id)
        continue;
    end
    distance_xy = norm(position(1:2) - zone.centroid_xy(:)');
    if distance_xy < radius
        if isnan(spray_state.dwell_start_time(zone.id))
            spray_state.dwell_start_time(zone.id) = t;
        end
        dwell_time = t - spray_state.dwell_start_time(zone.id);
        if dwell_time >= dwell_required
            latency_s = t - zone.first_detection_time;
            event = struct("zone_id", zone.id, ...
                           "time_s", t, ...
                           "position", position, ...
                           "latency_s", latency_s, ...
                           "dwell_s", dwell_time);
            spray_state.events(end + 1) = event;
            spray_state.latency_by_zone(zone.id) = latency_s;
            spray_state.sprayed(zone.id) = true;
            treatment = mark_sprayed_cells(treatment, zone.centroid_xy, radius);
            spray_state.dwell_start_time(zone.id) = NaN;
        end
    else
        spray_state.dwell_start_time(zone.id) = NaN;
    end
end

spray_events = spray_state.events;
latency_by_zone = spray_state.latency_by_zone;
end

function spray_state = initialize_spray_state(plan)
spray_state = struct();
if isfield(plan, "zones") && ~isempty(plan.zones)
    zone_ids = [plan.zones.id];
elseif isfield(plan, "zone_order")
    zone_ids = plan.zone_order;
else
    zone_ids = [];
end
spray_state.zone_order = zone_ids;   % retained for reference / back-compat
max_zone_id = max([0, zone_ids]);
% Per-zone state, indexed by zone id, so zones are tracked independently.
spray_state.sprayed = false(1, max_zone_id);
spray_state.dwell_start_time = nan(1, max_zone_id);
spray_state.events = struct("zone_id", {}, "time_s", {}, "position", {}, "latency_s", {}, "dwell_s", {});
spray_state.latency_by_zone = nan(1, max_zone_id);
end

function position = extract_position(uav_state)
if isfield(uav_state, "position")
    position = uav_state.position(:)';
elseif isfield(uav_state, "pos")
    position = uav_state.pos(:)';
elseif isfield(uav_state, "x")
    state = uav_state.x(:)';
    position = state(1:3);
else
    error("spray_executor:MissingPosition", "uav_state must contain position, pos, or x.");
end
end

function zone = find_zone_by_id(zones, zone_id)
idx = find([zones.id] == zone_id, 1);
if isempty(idx)
    error("spray_executor:MissingZone", "Zone id %d was not found in plan.zones.", zone_id);
end
zone = zones(idx);
end

function treatment = mark_sprayed_cells(treatment, centroid_xy, radius_m)
if ~isfield(treatment, "sprayed") || isempty(treatment.sprayed)
    return;
end
[rows, cols] = size(treatment.sprayed);
[col_grid, row_grid] = meshgrid(1:cols, 1:rows);
x = (col_grid - 0.5) * treatment.resolution_m;
y = (row_grid - 0.5) * treatment.resolution_m;
inside = hypot(x - centroid_xy(1), y - centroid_xy(2)) <= radius_m;
treatment.sprayed(inside) = true;
end
